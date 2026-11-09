"""FastAPI REST application and pipeline orchestrator for SieveRAG.

Provides endpoints for video ingestion, health inspection, temporal vector search,
grounded multimodal RAG question-answering, and static media hosting.
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

import vlmsieve
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.config import Settings, get_settings
from backend.app.extractor import StoryboardVLMExtractor
from backend.app.indexer import EpisodeEmbedder, SieveRAGVectorIndex
from backend.app.ingestor import SieveVideoIngestor
from backend.app.rag import SieveRAGSynthesizer
from backend.app.schemas import (
    AskRequest,
    AskResponse,
    SearchRequest,
    VideoIngestionReceipt,
)

logger = logging.getLogger("sieverag.api")


def create_app(
    workspace_dir: str | Path | None = None,
    frontend_dir: str | Path | None = None,
    extractor: StoryboardVLMExtractor | None = None,
    index: SieveRAGVectorIndex | None = None,
    synthesizer: SieveRAGSynthesizer | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    """Application factory constructing the SieveRAG FastAPI service.

    Args:
        workspace_dir: Optional root storage directory for video media and indexes.
        frontend_dir: Optional root directory for the static frontend UI.
        extractor: Optional injected StoryboardVLMExtractor.
        index: Optional injected SieveRAGVectorIndex.
        synthesizer: Optional injected SieveRAGSynthesizer.
        settings: Optional application settings.

    Returns:
        Configured FastAPI application instance.
    """
    app_settings = settings or get_settings()
    storage_dir = Path(workspace_dir).resolve() if workspace_dir else app_settings.STORAGE_DIR
    storage_dir.mkdir(parents=True, exist_ok=True)

    frontend_path = (
        Path(frontend_dir).resolve()
        if frontend_dir
        else (Path(__file__).resolve().parent.parent.parent / "frontend")
    )
    frontend_path.mkdir(parents=True, exist_ok=True)

    app = FastAPI(
        title="SieveRAG Backend API",
        version="0.3.0",
        description="Temporal Video RAG System powered by vlm-sieve, Google Gemini, and Pinecone.",
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Static Media Serving for Storyboards & Audio Slices (/media/...)
    app.mount("/media", StaticFiles(directory=str(storage_dir)), name="media")

    # Static UI Serving (/static/...)
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    # Pipeline Components
    app_extractor = extractor or StoryboardVLMExtractor(settings=app_settings)
    app_index = index or SieveRAGVectorIndex(
        embedder=EpisodeEmbedder(settings=app_settings),
        local_index_path=storage_dir / "vector_index.json",
        settings=app_settings,
    )
    app_synthesizer = synthesizer or SieveRAGSynthesizer(
        index=app_index,
        workspace_dir=storage_dir,
        settings=app_settings,
    )

    # ---------------------------------------------------------
    # REST Endpoints
    # ---------------------------------------------------------

    @app.get("/")
    def serve_frontend_ui() -> FileResponse:
        """Serve the Command Center UI single page app."""
        index_file = frontend_path / "index.html"
        if not index_file.is_file():
            raise HTTPException(status_code=404, detail="Frontend index.html not found.")
        return FileResponse(index_file)

    @app.get("/api/health")
    def health_check() -> dict[str, Any]:
        """Health check and engine status."""
        has_gemini = bool(app_settings.GEMINI_API_KEY) or (app_extractor.client is not None)
        return {
            "status": "ok",
            "vlmsieve_version": getattr(vlmsieve, "__version__", "0.3.0"),
            "target_vlm_default": app_settings.DEFAULT_TARGET_VLM,
            "gemini_configured": has_gemini,
            "vector_backend": "pinecone" if app_index.has_pinecone else "local",
            "indexed_episodes_count": len(app_index.local_store),
            "storage_dir": str(storage_dir),
        }

    @app.get("/api/stats")
    def get_global_stats() -> dict[str, Any]:
        """Calculate and return aggregate corpus telemetry across all ingested videos."""
        total_videos = 0
        total_duration_sec = 0.0
        total_raw_frames_inspected = 0
        total_storyboards_emitted = 0
        total_disturbances_suppressed = 0
        total_estimated_direct_tokens = 0
        total_actual_sieve_tokens = 0
        total_qwen_vl_patch_tokens = 0
        total_qwen7b_kv_cache_mb = 0.0

        for receipt_file in storage_dir.glob("*/ingestion_receipt.json"):
            try:
                data = json.loads(receipt_file.read_text(encoding="utf-8"))
                total_videos += 1
                total_duration_sec += float(data.get("duration_sec", 0.0))
                total_raw_frames_inspected += int(data.get("raw_frames_inspected", 0))
                total_storyboards_emitted += int(data.get("storyboards_emitted", 0))
                total_disturbances_suppressed += int(data.get("disturbances_suppressed", 0))
                total_estimated_direct_tokens += int(data.get("estimated_direct_video_tokens", 0))
                total_actual_sieve_tokens += int(data.get("actual_sieve_rag_tokens", 0))

                for ep in data.get("episodes", []):
                    total_qwen_vl_patch_tokens += int(ep.get("qwen_vl_patch_tokens", 0))
                    total_qwen7b_kv_cache_mb += float(ep.get("qwen7b_kv_cache_mb", 0.0))
            except Exception as exc:
                logger.warning("Could not read receipt for stats: %s (%s)", receipt_file, exc)

        if total_estimated_direct_tokens > 0:
            saved = max(0, total_estimated_direct_tokens - total_actual_sieve_tokens)
            overall_tokens_saved_pct = round((saved / float(total_estimated_direct_tokens)) * 100.0, 2)
        else:
            overall_tokens_saved_pct = 0.0

        return {
            "total_videos": total_videos,
            "total_duration_sec": round(total_duration_sec, 2),
            "total_raw_frames_inspected": total_raw_frames_inspected,
            "total_storyboards_emitted": total_storyboards_emitted,
            "total_disturbances_suppressed": total_disturbances_suppressed,
            "total_estimated_direct_tokens": total_estimated_direct_tokens,
            "total_actual_sieve_tokens": total_actual_sieve_tokens,
            "overall_tokens_saved_pct": overall_tokens_saved_pct,
            "total_qwen_vl_patch_tokens": total_qwen_vl_patch_tokens,
            "total_qwen7b_kv_cache_mb": round(total_qwen7b_kv_cache_mb, 2),
            "indexed_episodes_count": len(app_index.local_store),
            "vector_backend": "pinecone" if app_index.has_pinecone else "local",
        }

    @app.post("/api/videos/ingest")
    async def ingest_video_endpoint(
        file: UploadFile | None = File(default=None),
        local_video_path: str | None = Form(default=None),
        target_vlm: str = Form(default="gemini"),
        run_vlm_extraction: bool = Form(default=True),
        index_episodes: bool = Form(default=True),
    ) -> dict[str, Any]:
        """Ingest a video via file upload or local disk path.

        Runs vlm-sieve motion gating, storyboard generation, audio slicing,
        multimodal Gemini extraction, and vector indexation.
        """
        if file is None and not local_video_path:
            raise HTTPException(
                status_code=400,
                detail="Either 'file' upload or 'local_video_path' must be provided.",
            )

        video_id = f"vid_{uuid.uuid4().hex[:12]}"
        video_workspace = storage_dir / video_id
        video_workspace.mkdir(parents=True, exist_ok=True)

        # 1. Resolve and copy source video file into workspace for /media/ streaming
        if file is not None:
            filename = file.filename or "uploaded_video.mp4"
            src_video_path = video_workspace / filename
            content = await file.read()
            src_video_path.write_bytes(content)
        else:
            local_path = Path(local_video_path).resolve()  # type: ignore[arg-type]
            if not local_path.is_file():
                raise HTTPException(
                    status_code=404,
                    detail=f"Local video file not found: {local_video_path}",
                )
            dest_video_path = video_workspace / local_path.name
            if local_path != dest_video_path:
                shutil.copy2(local_path, dest_video_path)
            src_video_path = dest_video_path

        # 2. Run Phase 1 Ingestion
        ingestor = SieveVideoIngestor(
            settings=app_settings,
            target_vlm=target_vlm,
        )

        try:
            receipt_dict, events, disturbances, savings = ingestor.ingest_video(
                video_path=src_video_path,
                output_dir=video_workspace,
                video_id=video_id,
            )
        except Exception as exc:
            logger.error("Error during video ingestion: %s", exc, exc_info=True)
            raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

        receipt = VideoIngestionReceipt.model_validate(receipt_dict)

        # 3. Multimodal VLM Extraction (Phase 2)
        if run_vlm_extraction and receipt.episodes:
            can_call_gemini = (app_extractor.client is not None) or bool(app_extractor.api_key or app_settings.GEMINI_API_KEY)
            if can_call_gemini:
                try:
                    _maybe_receipt = app_extractor.extract_receipt(
                        receipt=receipt,
                        workspace_dir=video_workspace,
                        save_updated_receipt=True,
                    )
                except Exception as exc:
                    logger.warning("VLM extraction failed (%s); applying fallback extraction.", exc)
                    _apply_fallback_extraction(receipt, video_workspace)
            else:
                _apply_fallback_extraction(receipt, video_workspace)

        # 4. Vector Indexing (Phase 2)
        upserted_count = 0
        if index_episodes and receipt.episodes:
            try:
                upsert_res = app_index.upsert_episodes(receipt.episodes)
                upserted_count = upsert_res.get("upserted_count", len(receipt.episodes))
            except Exception as exc:
                logger.warning("Episode vector indexing failed: %s", exc)

        # Persist final updated receipt with video_url
        receipt_file = video_workspace / "ingestion_receipt.json"
        receipt_data = receipt.model_dump()
        receipt_data["upserted_vectors"] = upserted_count
        rel_vid = src_video_path.relative_to(storage_dir).as_posix()
        receipt_data["video_url"] = f"/media/{rel_vid.lstrip('/')}"
        receipt_file.write_text(json.dumps(receipt_data, indent=2), encoding="utf-8")

        return receipt_data

    @app.get("/api/videos")
    def list_ingested_videos() -> list[dict[str, Any]]:
        """List summary records of all ingested videos in storage."""
        summaries: list[dict[str, Any]] = []

        for receipt_file in storage_dir.glob("*/ingestion_receipt.json"):
            try:
                data = json.loads(receipt_file.read_text(encoding="utf-8"))
                vid_id = data.get("video_id", receipt_file.parent.name)
                v_url = data.get("video_url", "")
                if not v_url:
                    # Discover local video file if present
                    for vf in receipt_file.parent.glob("*.mp4"):
                        v_url = f"/media/{vf.relative_to(storage_dir).as_posix().lstrip('/')}"
                        break

                summaries.append({
                    "video_id": vid_id,
                    "video_filename": data.get("video_filename", ""),
                    "video_url": v_url,
                    "duration_sec": data.get("duration_sec", 0.0),
                    "storyboards_emitted": data.get("storyboards_emitted", 0),
                    "disturbances_suppressed": data.get("disturbances_suppressed", 0),
                    "tokens_saved_pct": data.get("tokens_saved_pct", 0.0),
                    "actual_sieve_rag_tokens": data.get("actual_sieve_rag_tokens", 0),
                    "estimated_direct_video_tokens": data.get("estimated_direct_video_tokens", 0),
                })
            except Exception as exc:
                logger.warning("Failed to parse receipt %s: %s", receipt_file, exc)

        summaries.sort(key=lambda x: str(x.get("video_id")), reverse=True)
        return summaries

    @app.get("/api/videos/{video_id}")
    def get_video_details(video_id: str) -> dict[str, Any]:
        """Retrieve full Ingestion Receipt for a specific video, enriched with media URLs."""
        video_workspace = storage_dir / video_id
        receipt_file = video_workspace / "ingestion_receipt.json"
        if not receipt_file.is_file():
            raise HTTPException(status_code=404, detail=f"Video receipt not found for ID: {video_id}")

        try:
            data = json.loads(receipt_file.read_text(encoding="utf-8"))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Failed to read receipt: {exc}")

        # Ensure video_url is populated
        v_url = data.get("video_url", "")
        if not v_url:
            for vf in video_workspace.glob("*.mp4"):
                v_url = f"/media/{vf.relative_to(storage_dir).as_posix().lstrip('/')}"
                break
        data["video_url"] = v_url

        # Enrich episodes with static media URLs
        for ep in data.get("episodes", []):
            sb_rel = ep.get("storyboard_rel_path", "")
            aud_rel = ep.get("audio_rel_path", "")
            if sb_rel:
                clean_sb = sb_rel.lstrip("/")
                ep["storyboard_url"] = f"/media/{clean_sb}" if clean_sb.startswith(video_id) else f"/media/{video_id}/{clean_sb}"
            else:
                ep["storyboard_url"] = ""

            if aud_rel:
                clean_aud = aud_rel.lstrip("/")
                ep["audio_url"] = f"/media/{clean_aud}" if clean_aud.startswith(video_id) else f"/media/{video_id}/{clean_aud}"
            else:
                ep["audio_url"] = ""

        return data

    @app.delete("/api/videos/{video_id}")
    def delete_video_endpoint(video_id: str) -> dict[str, Any]:
        """Delete a video, its storage directory, and all associated indexed vector embeddings."""
        video_workspace = storage_dir / video_id
        if not video_workspace.exists():
            raise HTTPException(status_code=404, detail=f"Video not found for ID: {video_id}")

        # 1. Delete vector embeddings
        removed_count = app_index.delete_video(video_id)

        # 2. Delete storage directory
        shutil.rmtree(video_workspace, ignore_errors=True)

        logger.info("Successfully deleted video '%s' (%d vectors removed)", video_id, removed_count)
        return {
            "deleted": True,
            "video_id": video_id,
            "removed_vectors": removed_count,
        }

    @app.post("/api/search")
    def search_episodes(request: SearchRequest) -> dict[str, Any]:
        """Perform semantic temporal vector search over indexed episodes."""
        hits = app_index.search(
            query=request.query,
            top_k=request.top_k,
            video_id=request.video_id,
            min_severity=request.min_severity,
            anomaly_only=request.anomaly_only,
        )

        # Enrich hits with media URLs
        for h in hits:
            vid = h.get("video_id", "")
            sb = h.get("storyboard_rel_path", "")
            aud = h.get("audio_rel_path", "")
            if sb and not h.get("storyboard_url"):
                clean_sb = sb.lstrip("/")
                h["storyboard_url"] = f"/media/{clean_sb}" if clean_sb.startswith(vid) else f"/media/{vid}/{clean_sb}"
            if aud and not h.get("audio_url"):
                clean_aud = aud.lstrip("/")
                h["audio_url"] = f"/media/{clean_aud}" if clean_aud.startswith(vid) else f"/media/{vid}/{clean_aud}"

        return {
            "query": request.query,
            "count": len(hits),
            "results": hits,
        }

    @app.post("/api/ask", response_model=AskResponse)
    def ask_rag_endpoint(request: AskRequest) -> AskResponse:
        """Ask a question about indexed videos and synthesize a grounded multimodal RAG response."""
        return app_synthesizer.answer_question(request)

    return app


def _apply_fallback_extraction(
    receipt: VideoIngestionReceipt,
    video_workspace: Path,
) -> None:
    """Populate baseline metadata when VLM extraction runs in offline/no-key mode."""
    for ep in receipt.episodes:
        if not ep.episode_title:
            ep.episode_title = f"Motion Episode #{ep.episode_index}"
        if not ep.narrative_summary:
            ep.narrative_summary = (
                f"Detected visual motion activity from {ep.start_time_sec:.2f}s to {ep.end_time_sec:.2f}s "
                f"across grid layout {ep.grid_used[0]}x{ep.grid_used[1]}."
            )

    receipt_file = video_workspace / "ingestion_receipt.json"
    receipt_file.write_text(json.dumps(receipt.model_dump(), indent=2), encoding="utf-8")


# Module-level default app instance for Uvicorn
app = create_app()



def _apply_fallback_extraction(
    receipt: VideoIngestionReceipt,
    video_workspace: Path,
) -> None:
    """Populate baseline metadata when VLM extraction runs in offline/no-key mode."""
    for ep in receipt.episodes:
        if not ep.episode_title:
            ep.episode_title = f"Motion Episode #{ep.episode_index}"
        if not ep.narrative_summary:
            ep.narrative_summary = (
                f"Detected visual motion activity from {ep.start_time_sec:.2f}s to {ep.end_time_sec:.2f}s "
                f"across grid layout {ep.grid_used[0]}x{ep.grid_used[1]}."
            )

    receipt_file = video_workspace / "ingestion_receipt.json"
    receipt_file.write_text(json.dumps(receipt.model_dump(), indent=2), encoding="utf-8")


# Module-level default app instance for Uvicorn
app = create_app()

