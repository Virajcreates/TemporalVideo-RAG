"""Temporal Video RAG Answer Synthesizer for SieveRAG.

Retrieves salient video episodes from vector index, constructs chronologically and
spatially grounded multimodal context (with storyboard image attachments), and invokes
Google Gemini to synthesize high-precision answers with exact temporal citations.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from backend.app.config import Settings, get_settings
from backend.app.indexer import SieveRAGVectorIndex
from backend.app.schemas import (
    AskRequest,
    AskResponse,
    RAGAnswerSchema,
    TemporalCitation,
)

logger = logging.getLogger("sieverag.rag")


def build_rag_prompt(question: str, hits: list[dict[str, Any]]) -> str:
    """Build a detailed grounded RAG prompt containing retrieved video episodes.

    Args:
        question: User query or surveillance question.
        hits: Ranked list of retrieved episode dictionaries from SieveRAGVectorIndex.

    Returns:
        Structured system/user context string for Gemini.
    """
    episodes_context_blocks: list[str] = []

    for idx, hit in enumerate(hits, 1):
        vid = hit.get("video_id", "unknown")
        ep_idx = hit.get("episode_index", idx)
        start_t = hit.get("start_time_sec", 0.0)
        end_t = hit.get("end_time_sec", 0.0)
        dur = max(0.0, end_t - start_t)
        title = hit.get("episode_title", "Untitled Episode")
        summary = hit.get("narrative_summary", "No summary provided.")
        actors = ", ".join(hit.get("actors", [])) or "None identified"
        objects = ", ".join(hit.get("objects_detected", [])) or "None detected"
        audio = hit.get("audio_transcript_and_cues", "No audio")
        severity = hit.get("severity", "low")
        anomaly = hit.get("anomaly_or_security_flag", False)
        sb_path = hit.get("storyboard_rel_path", "")
        grid_used = hit.get("grid_used", "2x3")

        # Format tile actions
        tile_actions_raw = hit.get("tile_actions", [])
        action_lines: list[str] = []
        if isinstance(tile_actions_raw, list):
            for t in tile_actions_raw:
                if isinstance(t, dict):
                    t_num = t.get("tile_number", "?")
                    rel_s = t.get("relative_time_sec", 0.0)
                    abs_s = t.get("absolute_time_sec", start_t + rel_s)
                    cells = ", ".join(t.get("grid_cells", [])) or "Full"
                    desc = t.get("action_description", "")
                    action_lines.append(
                        f"    - Tile #{t_num} | +{rel_s:.2f}s ({abs_s:.2f}s) | Grid: [{cells}] | Action: {desc}"
                    )

        tile_actions_str = "\n".join(action_lines) if action_lines else "    - No tile actions listed."

        block = f"""[RETRIEVED EPISODE #{idx}]
- Video ID: {vid} (Episode #{ep_idx})
- Time Interval: {start_t:.2f}s to {end_t:.2f}s ({dur:.2f}s duration)
- Active Grid: {grid_used}
- Title: {title}
- Summary: {summary}
- Actors: {actors}
- Key Objects: {objects}
- Audio Cues/Transcript: {audio}
- Security Anomaly: {anomaly} (Severity: {severity})
- Storyboard Relative Path: {sb_path}
- Chronological Tile Breakdown:
{tile_actions_str}"""
        episodes_context_blocks.append(block)

    all_context = "\n\n".join(episodes_context_blocks)

    return f"""You are SieveRAG, an expert AI Multimodal Surveillance & Video Analysis Intelligence Engine.
Your task is to answer the user's question using ONLY the retrieved video episode records and attached storyboard image grids below.

=== RETRIEVED VIDEO EPISODES CONTEXT ===
{all_context}

=== USER QUESTION ===
{question}

=== GROUNDING & RESPONSE INSTRUCTIONS ===
1. Base your answer strictly on the facts in the retrieved episodes and visual evidence in any attached storyboard images.
2. In your answer, explicitly cite exact timestamps, tile numbers (#1..#6), and spatial 4x4 grid cells (A1..D4) where relevant actions occurred.
3. If an action or object is not supported by the retrieved context, state clearly that it is not visible in the recorded video.
4. Set 'security_alert_triggered' to true if the cited evidence indicates security breaches, suspicious activity, intrusions, weapons/tools threats, falls, or safety hazards. Otherwise false.
5. Populate 'citations' with precise entries referencing the video_id, episode_index, start_time_sec, end_time_sec, specific tile_number, timestamp_sec, grid_cells, and factual evidence text.
6. Rate overall confidence as 'low', 'medium', or 'high'.
"""


class SieveRAGSynthesizer:
    """End-to-end RAG answer synthesizer for temporal video intelligence."""

    def __init__(
        self,
        index: SieveRAGVectorIndex,
        workspace_dir: str | Path | None = None,
        api_key: str | None = None,
        model_name: str | None = None,
        client: Any | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize SieveRAGSynthesizer.

        Args:
            index: SieveRAGVectorIndex instance for temporal semantic retrieval.
            workspace_dir: Base storage directory containing storyboards and audio slices.
            api_key: Optional Gemini API key.
            model_name: Target Gemini model (defaults to GEMINI_VLM_MODEL).
            client: Optional injected genai.Client instance for testing/mocking.
            settings: Optional application settings.
        """
        self.index = index
        self.settings = settings or get_settings()
        self.workspace_dir = Path(workspace_dir).resolve() if workspace_dir else self.settings.STORAGE_DIR
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self.model_name = model_name or self.settings.GEMINI_VLM_MODEL
        self.client = client

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        key = self.api_key or self.settings.GEMINI_API_KEY
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not configured for RAG synthesis. "
                "Pass api_key, set GEMINI_API_KEY, or inject a mock client."
            )
        self.client = genai.Client(api_key=key)
        return self.client

    def answer_question(self, request: AskRequest) -> AskResponse:
        """Answer a question using temporal video RAG with citations and visual grounding.

        Args:
            request: AskRequest instance containing question, filters, and top_k.

        Returns:
            AskResponse containing grounded answer, confidence, citations, and retrieved episodes.
        """
        t0 = time.perf_counter()

        # Step 1: Semantic Temporal Retrieval
        hits = self.index.search(
            query=request.question,
            top_k=request.top_k,
            video_id=request.video_id,
            min_severity=request.min_severity,
            anomaly_only=request.anomaly_only,
        )

        if not hits:
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return AskResponse(
                question=request.question,
                answer="No matching video episodes were found in the index for your query.",
                confidence="low",
                security_alert_triggered=False,
                citations=[],
                retrieved_episodes=[],
                prompt_tokens=0,
                output_tokens=0,
                latency_ms=latency_ms,
            )

        # Step 2: Build Multimodal Content (Storyboard JPEGs + Text Prompt)
        parts: list[Any] = []
        if request.include_storyboards and self.workspace_dir:
            # Attach top 3 retrieved storyboard images for visual verification
            for hit in hits[:3]:
                sb_rel = hit.get("storyboard_rel_path")
                if sb_rel:
                    sb_full = self.workspace_dir / sb_rel
                    if sb_full.is_file() and sb_full.stat().st_size > 0:
                        try:
                            parts.append(types.Part.from_bytes(data=sb_full.read_bytes(), mime_type="image/jpeg"))
                        except Exception as exc:
                            logger.warning("Could not read storyboard %s: %s", sb_full, exc)

        prompt_text = build_rag_prompt(request.question, hits)
        parts.append(types.Part.from_text(text=prompt_text))

        # Step 3: Synthesis via Gemini VLM or Deterministic Offline Fallback
        prompt_tokens = 0
        output_tokens = 0
        rag_answer: RAGAnswerSchema

        can_use_gemini = (self.client is not None) or bool(self.api_key or self.settings.GEMINI_API_KEY)

        if can_use_gemini:
            try:
                client = self._get_client()
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                        response_schema=RAGAnswerSchema,
                    ),
                )

                if hasattr(response, "parsed") and response.parsed is not None:
                    if isinstance(response.parsed, RAGAnswerSchema):
                        rag_answer = response.parsed
                    elif isinstance(response.parsed, dict):
                        rag_answer = RAGAnswerSchema.model_validate(response.parsed)
                    else:
                        rag_answer = RAGAnswerSchema.model_validate(response.parsed)
                elif hasattr(response, "text") and response.text:
                    raw_text = response.text.strip()
                    if raw_text.startswith("```"):
                        lines = raw_text.splitlines()
                        if lines and lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        raw_text = "\n".join(lines).strip()
                    rag_answer = RAGAnswerSchema.model_validate_json(raw_text)
                else:
                    raise ValueError("No parsed output or text returned from Gemini RAG synthesis.")

                usage = getattr(response, "usage_metadata", None)
                if usage:
                    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
                    output_tokens = getattr(usage, "candidates_token_count", 0) or 0

            except Exception as exc:
                logger.warning(
                    "Gemini RAG synthesis failed (%s); using deterministic offline fallback.",
                    exc,
                )
                rag_answer = self._deterministic_offline_fallback(request.question, hits)
        else:
            rag_answer = self._deterministic_offline_fallback(request.question, hits)

        # Step 4: Ensure URLs on all citations are clean and formatted (/media/...)
        self._enrich_citation_urls(rag_answer.citations, hits)

        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return AskResponse(
            question=request.question,
            answer=rag_answer.answer,
            confidence=rag_answer.confidence,
            security_alert_triggered=rag_answer.security_alert_triggered,
            citations=rag_answer.citations,
            retrieved_episodes=hits,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
        )

    def _deterministic_offline_fallback(
        self,
        question: str,
        hits: list[dict[str, Any]],
    ) -> RAGAnswerSchema:
        """Deterministic offline synthesizer used when no API key is provided."""
        top_hit = hits[0]
        vid = top_hit.get("video_id", "unknown")
        ep_idx = int(top_hit.get("episode_index", 1))
        start_t = float(top_hit.get("start_time_sec", 0.0))
        end_t = float(top_hit.get("end_time_sec", 0.0))
        title = top_hit.get("episode_title", "Visual Activity")
        summary = top_hit.get("narrative_summary", "")
        objects = top_hit.get("objects_detected", [])
        is_anomaly = bool(top_hit.get("anomaly_or_security_flag", False))
        severity = str(top_hit.get("severity", "low"))

        tile_actions = top_hit.get("tile_actions", [])
        first_action = tile_actions[0] if (isinstance(tile_actions, list) and tile_actions) else {}
        t_num = first_action.get("tile_number", 1) if isinstance(first_action, dict) else 1
        abs_ts = first_action.get("absolute_time_sec", start_t) if isinstance(first_action, dict) else start_t
        grid_cells = first_action.get("grid_cells", []) if isinstance(first_action, dict) else []
        action_desc = first_action.get("action_description", "") if isinstance(first_action, dict) else ""

        answer_parts = [
            f"Based on indexed video '{vid}' (Episode #{ep_idx}, {start_t:.2f}s to {end_t:.2f}s, '{title}'):",
            f"{summary}.",
        ]
        if action_desc:
            grid_suffix = f" in cell {','.join(grid_cells)}" if grid_cells else ""
            answer_parts.append(
                f"At Tile #{t_num} (+{abs_ts - start_t:.2f}s / {abs_ts:.2f}s{grid_suffix}), {action_desc}."
            )
        if objects:
            answer_parts.append(f"Identified objects: {', '.join(objects)}.")

        answer_text = " ".join(answer_parts)

        sb_rel = top_hit.get("storyboard_rel_path", "")
        audio_rel = top_hit.get("audio_rel_path", "")

        citation = TemporalCitation(
            video_id=vid,
            episode_index=ep_idx,
            start_time_sec=start_t,
            end_time_sec=end_t,
            tile_number=t_num,
            timestamp_sec=abs_ts,
            grid_cells=grid_cells,
            evidence=summary or action_desc or title,
            storyboard_url=f"/media/{sb_rel.lstrip('/')}" if sb_rel else "",
            audio_url=f"/media/{audio_rel.lstrip('/')}" if audio_rel else "",
        )

        return RAGAnswerSchema(
            answer=answer_text,
            confidence="high" if top_hit.get("score", 0.0) >= 0.3 else "medium",
            security_alert_triggered=is_anomaly or (severity == "high"),
            citations=[citation],
        )

    def _enrich_citation_urls(
        self,
        citations: list[TemporalCitation],
        hits: list[dict[str, Any]],
    ) -> None:
        """Ensure all citation storyboard_url and audio_url point to /media/... endpoints."""
        hit_map = {
            (h.get("video_id"), int(h.get("episode_index", 0))): h
            for h in hits
        }

        for c in citations:
            matched_hit = hit_map.get((c.video_id, c.episode_index)) or (hits[0] if hits else None)
            if matched_hit:
                sb_rel = matched_hit.get("storyboard_rel_path", "")
                aud_rel = matched_hit.get("audio_rel_path", "")

                if sb_rel and (not c.storyboard_url or not c.storyboard_url.startswith(("/media/", "http"))):
                    c.storyboard_url = f"/media/{sb_rel.lstrip('/')}"

                if aud_rel and (not c.audio_url or not c.audio_url.startswith(("/media/", "http"))):
                    c.audio_url = f"/media/{aud_rel.lstrip('/')}"

