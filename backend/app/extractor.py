"""Multimodal VLM structured extractor for SieveRAG.

Processes composite storyboard images and synchronized audio slices with
Google Gemini (via google-genai SDK) to extract structured, spatially and temporally
grounded episode records.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from backend.app.config import Settings, get_settings
from backend.app.schemas import (
    EpisodeExtractionSchema,
    IndexedEpisodeRecord,
    VideoIngestionReceipt,
)

logger = logging.getLogger("sieverag.extractor")


def build_extraction_prompt(record: IndexedEpisodeRecord) -> str:
    """Build a comprehensive multimodal extraction prompt for Gemini.

    Dynamically adapts to the episode's duration, active grid layout,
    and exact keyframe timestamps.

    Args:
        record: The indexed episode record containing sieve metadata.

    Returns:
        Structured system/user prompt string for Gemini VLM.
    """
    rows, cols = record.grid_used
    num_tiles = (
        len(record.keyframe_times_sec)
        if record.keyframe_times_sec
        else (rows * cols)
    )

    # Format keyframe timestamps
    kf_lines: list[str] = []
    if record.keyframe_times_sec:
        for idx, abs_t in enumerate(record.keyframe_times_sec, 1):
            rel_t = max(0.0, abs_t - record.start_time_sec)
            kf_lines.append(f"  - Tile #{idx}: +{rel_t:.2f}s (abs {abs_t:.2f}s)")
    else:
        # Interpolate estimated times if keyframe list was not explicitly populated
        step = record.duration_sec / max(1, num_tiles - 1) if num_tiles > 1 else 0.0
        for idx in range(1, num_tiles + 1):
            rel_t = (idx - 1) * step
            abs_t = record.start_time_sec + rel_t
            kf_lines.append(f"  - Tile #{idx}: +{rel_t:.2f}s (abs {abs_t:.2f}s)")

    timestamps_block = "\n".join(kf_lines)

    audio_status = (
        "Synchronized mono MP3 audio slice is attached to this request."
        if record.audio_rel_path
        else "No audio stream is present for this video episode."
    )

    return f"""You are an expert Multimodal Video Intelligence and Surveillance Analysis VLM.
You are analyzing an active motion episode extracted by vlm-sieve from video '{record.video_filename}'.

=== EPISODE TEMPORAL METADATA ===
- Time Window: {record.start_time_sec:.2f}s to {record.end_time_sec:.2f}s ({record.duration_sec:.2f}s total duration)
- Active Grid Layout: {rows}x{cols} ({num_tiles} populated keyframe tiles)
- Chronological Tile Timestamps:
{timestamps_block}
- Audio Status: {audio_status}

=== STORYBOARD SPECIFICATIONS & READING RULES ===
1. Each tile has a top-left badge `#tile | +offset_sec` and spatial 4x4 grid coordinates (`A1..D4`, where rows are `A, B, C, D` top-to-bottom and columns are `1, 2, 3, 4` left-to-right).
2. Inspect every keyframe tile across the trajectory to identify tools, handheld objects, clothing, and actions from multiple angles. Do not guess from a single flat angle if adjacent tiles reveal the object's 3D profile.
3. Only generate `tile_actions` for the actual populated tiles (1..{num_tiles}). Never invent actions for empty/unpopulated tile slots.
4. If an audio clip is attached, transcribe any speech verbatim and note acoustic events (doorbell chimes, footsteps, knocks, alarms, glass breaking) in `audio_transcript_and_cues`. If no audio is attached, set `audio_transcript_and_cues` to 'No audio stream or silent episode.'
5. Set `anomaly_or_security_flag` to true if you detect suspicious activity, weapon/tool threats, forced entry, intrusion, vandalism, falls, or safety violations. Otherwise set to false.
6. Rate `severity` as 'low', 'medium', or 'high'.

Provide your complete, factual analysis matching the requested response schema.
"""


class StoryboardVLMExtractor:
    """Multimodal Vision-Language Model extractor powered by Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        client: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize StoryboardVLMExtractor.

        Args:
            api_key: Optional Gemini API key. If omitted, loaded from settings/env.
            model_name: Target Gemini model name (defaults to GEMINI_VLM_MODEL).
            client: Optional pre-configured genai.Client instance (useful for unit testing / mocking).
            settings: Optional application settings.
        """
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self.model_name = model_name or self.settings.GEMINI_VLM_MODEL
        self.client = client

    def _get_client(self) -> Any:
        """Lazily initialize and return the google-genai Client."""
        if self.client is not None:
            return self.client

        key = self.api_key or self.settings.GEMINI_API_KEY
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY environment variable "
                "or pass api_key to StoryboardVLMExtractor."
            )
        self.client = genai.Client(api_key=key)
        return self.client

    def extract_episode(
        self,
        record: IndexedEpisodeRecord,
        workspace_dir: str | Path,
    ) -> IndexedEpisodeRecord:
        """Execute multimodal structured extraction for a single episode record.

        Args:
            record: The IndexedEpisodeRecord to enrich.
            workspace_dir: Base directory containing storyboards and audio slices.

        Returns:
            The mutated record with populated extraction fields and token telemetry.

        Raises:
            FileNotFoundError: If the storyboard image is missing on disk.
        """
        base_dir = Path(workspace_dir).resolve()
        storyboard_path = base_dir / record.storyboard_rel_path

        if not storyboard_path.is_file():
            raise FileNotFoundError(f"Storyboard image not found: {storyboard_path}")

        # Load composite storyboard JPEG bytes
        img_bytes = storyboard_path.read_bytes()
        parts: list[Any] = [types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg")]

        # Load audio MP3 slice if present
        if record.audio_rel_path:
            audio_path = base_dir / record.audio_rel_path
            if audio_path.is_file() and audio_path.stat().st_size > 0:
                audio_bytes = audio_path.read_bytes()
                parts.append(types.Part.from_bytes(data=audio_bytes, mime_type="audio/mpeg"))
            else:
                logger.debug("Audio file %s not found or empty; extracting visual only", audio_path)

        # Build dynamic prompt
        prompt = build_extraction_prompt(record)
        parts.append(types.Part.from_text(text=prompt))

        logger.info(
            "Invoking Gemini VLM (%s) for episode #%d in video '%s'",
            self.model_name,
            record.episode_index,
            record.video_id,
        )

        client = self._get_client()
        response = client.models.generate_content(
            model=self.model_name,
            contents=parts,
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
                response_schema=EpisodeExtractionSchema,
            ),
        )

        # Parse structured extraction result
        extraction: EpisodeExtractionSchema
        if hasattr(response, "parsed") and response.parsed is not None:
            if isinstance(response.parsed, EpisodeExtractionSchema):
                extraction = response.parsed
            elif isinstance(response.parsed, dict):
                extraction = EpisodeExtractionSchema.model_validate(response.parsed)
            else:
                extraction = EpisodeExtractionSchema.model_validate(response.parsed)
        elif hasattr(response, "text") and response.text:
            raw_text = response.text.strip()
            # Strip markdown json code block if present
            if raw_text.startswith("```"):
                lines = raw_text.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_text = "\n".join(lines).strip()
            extraction = EpisodeExtractionSchema.model_validate_json(raw_text)
        else:
            raise ValueError("Gemini response contained neither parsed structured output nor text content.")

        # Extract token telemetry
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) or record.sieve_input_tokens
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        record.apply_extraction(
            extraction=extraction,
            input_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

        logger.info(
            "Episode #%d extraction complete: '%s' (severity=%s, in_tokens=%d, out_tokens=%d)",
            record.episode_index,
            record.episode_title,
            record.severity,
            record.sieve_input_tokens,
            record.sieve_output_tokens,
        )

        return record

    def extract_receipt(
        self,
        receipt: VideoIngestionReceipt,
        workspace_dir: str | Path,
        save_updated_receipt: bool = True,
    ) -> VideoIngestionReceipt:
        """Run multimodal VLM extraction over all episodes in a VideoIngestionReceipt.

        Updates cumulative token counts, recalculates savings percentages,
        and optionally overwrites ingestion_receipt.json in workspace_dir.

        Args:
            receipt: The VideoIngestionReceipt from Phase 1 ingestion.
            workspace_dir: Directory containing receipt and media assets.
            save_updated_receipt: If True, writes updated receipt to disk.

        Returns:
            Enriched VideoIngestionReceipt instance.
        """
        base_dir = Path(workspace_dir).resolve()

        for ep in receipt.episodes:
            self.extract_episode(record=ep, workspace_dir=base_dir)

        # Update actual sieve RAG tokens with actual prompt tokens consumed
        receipt.actual_sieve_rag_tokens = sum(ep.sieve_input_tokens for ep in receipt.episodes)

        if receipt.estimated_direct_video_tokens > 0:
            saved = max(0, receipt.estimated_direct_video_tokens - receipt.actual_sieve_rag_tokens)
            receipt.tokens_saved_pct = round(
                (saved / float(receipt.estimated_direct_video_tokens)) * 100.0, 2
            )
        else:
            receipt.tokens_saved_pct = 0.0

        if save_updated_receipt:
            receipt_path = base_dir / "ingestion_receipt.json"
            receipt_path.write_text(
                json.dumps(receipt.model_dump(), indent=2),
                encoding="utf-8",
            )
            logger.info("Persisted updated extraction receipt to %s", receipt_path)

        return receipt

