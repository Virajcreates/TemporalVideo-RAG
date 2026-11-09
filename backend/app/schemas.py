"""Pydantic v2 schemas for SieveRAG ingestion, extraction, and indexing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from vlmsieve.models import VisualEvent


class TileAction(BaseModel):
    """Fine-grained temporal action and spatial coordinates for a single storyboard tile."""

    model_config = ConfigDict(extra="ignore")

    tile_number: int = Field(
        ...,
        ge=1,
        le=6,
        description="Index of the storyboard tile in chronological reading order (1 to 6)",
    )
    relative_time_sec: float = Field(
        ...,
        description="Offset in seconds from the start of the current episode (e.g. 1.25)",
    )
    absolute_time_sec: float = Field(
        ...,
        description="Absolute timestamp in seconds from the beginning of the source video",
    )
    grid_cells: list[str] = Field(
        default_factory=list,
        description="Spatial 4x4 grid cell coordinates where the subject or motion occurred (e.g. ['B2', 'C3'])",
    )
    action_description: str = Field(
        ...,
        description="Concise description of the specific action, movement, or state visible in this tile",
    )

    @property
    def action(self) -> str:
        """Convenience property alias for action_description."""
        return self.action_description


class EpisodeExtractionSchema(BaseModel):
    """Structured extraction schema passed to Gemini VLM as response_schema."""

    model_config = ConfigDict(extra="ignore")

    episode_title: str = Field(
        default="",
        description="Concise headline or title describing the visual event episode",
    )
    narrative_summary: str = Field(
        default="",
        description="Comprehensive narrative paragraph detailing what occurred across the storyboard tiles",
    )
    actors: list[str] = Field(
        default_factory=list,
        description="List of identifiable entities, persons, animals, or vehicles involved in the episode",
    )
    objects_detected: list[str] = Field(
        default_factory=list,
        description="List of significant physical objects, tools, containers, or assets observed",
    )
    audio_transcript_and_cues: str = Field(
        default="",
        description="Speech transcript, dialogue, or notable auditory events (e.g., footsteps, glass shatter, siren) during the episode slice",
    )
    anomaly_or_security_flag: bool = Field(
        default=False,
        description="True if an anomaly, safety violation, security breach, intrusion, or suspicious activity is detected",
    )
    severity: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Assessed threat or operational severity level: 'low', 'medium', or 'high'",
    )
    tile_actions: list[TileAction] = Field(
        default_factory=list,
        description="Chronological, tile-by-tile action descriptions grounded with relative/absolute time and spatial grid cells",
    )


class IndexedEpisodeRecord(EpisodeExtractionSchema):
    """Unified record combining vlm-sieve physical metadata with VLM extraction schema."""

    model_config = ConfigDict(extra="ignore")

    # Source & Sieve Metadata
    video_id: str = Field(
        ...,
        description="Unique identifier for the ingested source video (UUID or slug)",
    )
    video_filename: str = Field(
        ...,
        description="Original filename of the ingested video file",
    )
    event_id: int = Field(
        ...,
        description="Sequential event counter emitted by vlm-sieve",
    )
    episode_index: int = Field(
        ...,
        description="1-based chronological episode index within this video",
    )
    start_time_sec: float = Field(
        ...,
        description="Episode start timestamp in seconds within the source video",
    )
    end_time_sec: float = Field(
        ...,
        description="Episode end timestamp in seconds within the source video",
    )
    duration_sec: float = Field(
        ...,
        description="Total duration of this episode in seconds (end_time_sec - start_time_sec)",
    )
    keyframe_times_sec: list[float] = Field(
        default_factory=list,
        description="Timestamps in seconds of the selected keyframes packed into the storyboard tiles",
    )
    roi_bbox: dict[str, Any] = Field(
        default_factory=dict,
        description="Active Region of Interest bounding box {'x': int, 'y': int, 'width': int, 'height': int}",
    )
    storyboard_rel_path: str = Field(
        ...,
        description="Relative file path to the saved composite storyboard image (JPEG)",
    )
    audio_rel_path: str | None = Field(
        default=None,
        description="Relative file path to the sliced mono MP3 audio file, or None if video had no audio",
    )
    sieve_input_tokens: int = Field(
        default=1120,
        description="VLM input token budget consumed by the composite storyboard image grid",
    )
    sieve_output_tokens: int = Field(
        default=0,
        description="VLM output tokens consumed during structured extraction",
    )
    grid_used: tuple[int, int] = Field(
        default=(2, 3),
        description="Active grid layout (rows, cols) chosen for this episode storyboard",
    )
    qwen_vl_patch_tokens: int = Field(
        default=0,
        description="Estimated token cost under Qwen-VL patch tokenization",
    )
    qwen7b_kv_cache_mb: float = Field(
        default=0.0,
        description="Estimated KV cache memory footprint in MB for Qwen-7B",
    )

    def to_extraction_schema(self) -> EpisodeExtractionSchema:
        """Extract the pure EpisodeExtractionSchema component."""
        return EpisodeExtractionSchema(
            episode_title=self.episode_title,
            narrative_summary=self.narrative_summary,
            actors=list(self.actors),
            objects_detected=list(self.objects_detected),
            audio_transcript_and_cues=self.audio_transcript_and_cues,
            anomaly_or_security_flag=self.anomaly_or_security_flag,
            severity=self.severity,
            tile_actions=list(self.tile_actions),
        )

    def apply_extraction(
        self,
        extraction: EpisodeExtractionSchema,
        input_tokens: int | None = None,
        output_tokens: int = 0,
    ) -> None:
        """Update extraction fields from a completed VLM extraction result."""
        self.episode_title = extraction.episode_title
        self.narrative_summary = extraction.narrative_summary
        self.actors = list(extraction.actors)
        self.objects_detected = list(extraction.objects_detected)
        self.audio_transcript_and_cues = extraction.audio_transcript_and_cues
        self.anomaly_or_security_flag = extraction.anomaly_or_security_flag
        self.severity = extraction.severity
        self.tile_actions = list(extraction.tile_actions)
        if input_tokens is not None and input_tokens > 0:
            self.sieve_input_tokens = input_tokens
        if output_tokens > 0:
            self.sieve_output_tokens = output_tokens

    def to_embedding_text(self) -> str:
        """Construct a rich, deterministic semantic search string combining title, summary, objects, and tile actions."""
        timeline_str = " | ".join(
            f"[Tile #{t.tile_number} +{t.relative_time_sec:.2f}s @ {','.join(t.grid_cells)}] {t.action}"
            for t in self.tile_actions
        )
        return (
            f"Title: {self.episode_title}\n"
            f"Summary: {self.narrative_summary}\n"
            f"Actors: {', '.join(self.actors)}\n"
            f"Objects: {', '.join(self.objects_detected)}\n"
            f"Audio/Transcript: {self.audio_transcript_and_cues}\n"
            f"Severity: {self.severity} (Anomaly: {self.anomaly_or_security_flag})\n"
            f"Timeline ({self.start_time_sec:.2f}s to {self.end_time_sec:.2f}s): {timeline_str}"
        )

    def to_vector_metadata(self) -> dict[str, Any]:
        """Return a flat dictionary of Pinecone-compatible primitives (str, int, float, bool, list[str]).

        Crucially avoids nested dicts or None values, which are rejected by Pinecone metadata.
        """
        return {
            "video_id": str(self.video_id),
            "video_filename": str(self.video_filename),
            "event_id": str(self.event_id),
            "episode_index": int(self.episode_index),
            "start_time_sec": float(self.start_time_sec),
            "end_time_sec": float(self.end_time_sec),
            "duration_sec": float(self.duration_sec),
            "storyboard_rel_path": str(self.storyboard_rel_path),
            "audio_rel_path": str(self.audio_rel_path or ""),
            "episode_title": str(self.episode_title),
            "narrative_summary": str(self.narrative_summary),
            "actors": [str(a) for a in self.actors],
            "objects_detected": [str(o) for o in self.objects_detected],
            "audio_transcript_and_cues": str(self.audio_transcript_and_cues),
            "anomaly_or_security_flag": bool(self.anomaly_or_security_flag),
            "severity": str(self.severity),
            "grid_used_str": f"{self.grid_used[0]}x{self.grid_used[1]}",
            "tile_actions_json": json.dumps([t.model_dump() for t in self.tile_actions]),
        }

    @classmethod
    def from_sieve_event(
        cls,
        event: VisualEvent,
        video_id: str,
        video_filename: str,
        storyboard_rel_path: str,
        audio_rel_path: str | None = None,
        extraction: EpisodeExtractionSchema | None = None,
        sieve_input_tokens: int = 1120,
        sieve_output_tokens: int = 0,
    ) -> IndexedEpisodeRecord:
        """Factory method to construct an IndexedEpisodeRecord from a vlm-sieve VisualEvent."""
        duration = round(event.end_time_sec - event.start_time_sec, 3)
        roi_dict = {
            "x": int(event.roi_bbox.x),
            "y": int(event.roi_bbox.y),
            "width": int(event.roi_bbox.width),
            "height": int(event.roi_bbox.height),
        }

        # Normalize relative path using forward slashes for cross-platform consistency
        norm_storyboard_path = Path(storyboard_rel_path).as_posix()
        norm_audio_path = Path(audio_rel_path).as_posix() if audio_rel_path else None

        # vlm-sieve 0.3.0 attributes
        grid_used = getattr(event, "grid_used", (2, 3))
        qwen_tokens = 0
        qwen_mb = 0.0
        if hasattr(event, "provider_token_estimates") and callable(event.provider_token_estimates):
            try:
                estimates = event.provider_token_estimates()
                qwen_tokens = int(estimates.get("qwen_vl_tokens", 0))
                qwen_mb = float(estimates.get("qwen7b_kv_cache_mb", 0.0))
            except Exception:
                pass

        base_data: dict[str, Any] = {
            "video_id": video_id,
            "video_filename": video_filename,
            "event_id": event.event_id,
            "episode_index": getattr(event, "episode_index", event.event_id),
            "start_time_sec": round(event.start_time_sec, 3),
            "end_time_sec": round(event.end_time_sec, 3),
            "duration_sec": duration,
            "keyframe_times_sec": list(getattr(event, "keyframe_times_sec", [])),
            "roi_bbox": roi_dict,
            "storyboard_rel_path": norm_storyboard_path,
            "audio_rel_path": norm_audio_path,
            "sieve_input_tokens": sieve_input_tokens,
            "sieve_output_tokens": sieve_output_tokens,
            "grid_used": grid_used,
            "qwen_vl_patch_tokens": qwen_tokens,
            "qwen7b_kv_cache_mb": qwen_mb,
        }

        if extraction is not None:
            base_data.update(extraction.model_dump())

        return cls(**base_data)


class VideoIngestionReceipt(BaseModel):
    """Complete summary and audit receipt for an ingested video in SieveRAG."""

    model_config = ConfigDict(extra="ignore")

    video_id: str = Field(
        ...,
        description="Unique identifier for the processed video",
    )
    video_filename: str = Field(
        ...,
        description="Source video filename",
    )
    duration_sec: float = Field(
        ...,
        description="Total duration of the input video in seconds",
    )
    raw_frames_inspected: int = Field(
        ...,
        description="Total raw frames processed by the motion gatekeeper",
    )
    sampled_frames_decoded: int = Field(
        ...,
        description="Frames actually decoded and evaluated at sample_fps",
    )
    storyboards_emitted: int = Field(
        ...,
        description="Count of salient 2x3 storyboard image grids generated",
    )
    disturbances_suppressed: int = Field(
        ...,
        description="Global camera shake or flash events discarded without emitting storyboards",
    )
    processing_wall_sec: float = Field(
        ...,
        description="End-to-end CPU wall-clock processing time in seconds",
    )
    estimated_direct_video_tokens: int = Field(
        ...,
        description="Estimated token cost of sending entire raw video directly (duration * 91.5 tokens/sec)",
    )
    actual_sieve_rag_tokens: int = Field(
        ...,
        description="Actual token cost under SieveRAG (storyboards_emitted * tokens_per_image)",
    )
    tokens_saved_pct: float = Field(
        ...,
        description="Percentage token savings achieved relative to direct video ingestion",
    )
    episodes: list[IndexedEpisodeRecord] = Field(
        default_factory=list,
        description="Indexed episode records prepared for Phase 2 VLM indexing",
    )
    disturbances: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Metadata records of suppressed camera shake / exposure disturbance events",
    )


class TemporalCitation(BaseModel):
    """Fine-grained temporal and spatial grounding citation pointing to a video episode."""

    model_config = ConfigDict(extra="ignore")

    video_id: str = Field(..., description="Identifier of the cited video")
    episode_index: int = Field(..., description="Episode index within the cited video")
    start_time_sec: float = Field(..., description="Episode start timestamp in seconds")
    end_time_sec: float = Field(..., description="Episode end timestamp in seconds")
    tile_number: int | None = Field(default=None, description="Specific storyboard tile number (1..6) if applicable")
    timestamp_sec: float | None = Field(default=None, description="Exact timestamp in seconds of observed event")
    grid_cells: list[str] = Field(default_factory=list, description="Spatial 4x4 grid cells where event was observed (e.g. ['B2', 'C3'])")
    evidence: str = Field(..., description="Direct factual observation or quote grounding this citation")
    storyboard_url: str = Field(default="", description="Web-accessible URL to the episode storyboard JPEG image")
    audio_url: str = Field(default="", description="Web-accessible URL to the episode audio MP3 slice")


class RAGAnswerSchema(BaseModel):
    """Structured answer format produced by Gemini during multimodal RAG synthesis."""

    model_config = ConfigDict(extra="ignore")

    answer: str = Field(
        ...,
        description="Comprehensive, grounded answer citing exact timestamps, tile numbers, and A1..D4 spatial grid coordinates.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="high",
        description="Model confidence level regarding the factual accuracy of the answer",
    )
    security_alert_triggered: bool = Field(
        default=False,
        description="True if the synthesized response identifies an active security, safety, or anomaly risk",
    )
    citations: list[TemporalCitation] = Field(
        default_factory=list,
        description="List of precise temporal and spatial evidence citations supporting the answer",
    )


class SearchRequest(BaseModel):
    """Semantic temporal vector search request."""

    model_config = ConfigDict(extra="ignore")

    query: str = Field(..., min_length=1, description="Natural language search query")
    top_k: int = Field(default=5, ge=1, le=25, description="Maximum number of episode hits to return")
    video_id: str | None = Field(default=None, description="Filter results to a specific video identifier")
    anomaly_only: bool = Field(default=False, description="Filter results strictly to flagged anomaly episodes")
    min_severity: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="Filter results by minimum severity tier",
    )


class AskRequest(BaseModel):
    """Temporal Video RAG synthesis query request."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., min_length=1, description="User question regarding video surveillance/content")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of top temporal episodes to retrieve as context")
    video_id: str | None = Field(default=None, description="Optional filter to scope retrieval to a single video")
    anomaly_only: bool = Field(default=False, description="Filter context strictly to flagged anomaly episodes")
    min_severity: Literal["low", "medium", "high"] | None = Field(
        default=None,
        description="Filter context by minimum severity level",
    )
    include_storyboards: bool = Field(
        default=True,
        description="Whether to attach top retrieved storyboard JPEGs directly to the Gemini VLM synthesis call",
    )


class AskResponse(BaseModel):
    """Grounded multimodal video RAG response with temporal citations and retrieved context."""

    model_config = ConfigDict(extra="ignore")

    question: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Synthesized grounded answer")
    confidence: Literal["low", "medium", "high"] = Field(..., description="Answer confidence rating")
    security_alert_triggered: bool = Field(..., description="Whether a security threat or safety alert was triggered")
    citations: list[TemporalCitation] = Field(default_factory=list, description="Grounding citations with timestamp and tile pointers")
    retrieved_episodes: list[dict[str, Any]] = Field(default_factory=list, description="Raw metadata for retrieved episode candidates")
    prompt_tokens: int = Field(default=0, description="VLM prompt tokens consumed during synthesis")
    output_tokens: int = Field(default=0, description="VLM response tokens consumed during synthesis")
    latency_ms: float = Field(default=0.0, description="Total synthesis wall-clock latency in milliseconds")


