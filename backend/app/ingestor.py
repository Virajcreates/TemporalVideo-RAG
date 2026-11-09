"""Core ingestion and audio slicing engine for SieveRAG.

Leverages vlm-sieve (>=0.2.1) to sample video at motion peaks, auto-crop active
ROIs, render 4x4 spatial overlays, burn chronological timestamps, suppress camera
disturbances, and slice synchronized mono MP3 audio per emitted visual event.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import imageio_ffmpeg
from vlmsieve import SieveConfig, VideoSieve
from vlmsieve.models import DisturbanceRecord, TokenSavingsReport, VisualEvent

from backend.app.config import Settings, get_settings
from backend.app.schemas import IndexedEpisodeRecord, VideoIngestionReceipt

logger = logging.getLogger("sieverag.ingestor")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def get_ffmpeg_binary() -> str:
    """Resolve the FFmpeg executable binary.

    Prioritizes the self-contained static binary from imageio-ffmpeg,
    falling back to system PATH if unavailable.

    Returns:
        Absolute string path to the executable ffmpeg binary.

    Raises:
        RuntimeError: If no valid FFmpeg binary could be found.
    """
    try:
        binary = imageio_ffmpeg.get_ffmpeg_exe()
        if binary and Path(binary).is_file():
            return str(Path(binary).resolve())
    except Exception as exc:
        logger.warning("Failed to resolve imageio-ffmpeg binary: %s", exc)

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return str(Path(system_ffmpeg).resolve())

    raise RuntimeError(
        "FFmpeg executable not found. Please install imageio-ffmpeg or ensure ffmpeg is on system PATH."
    )


class SieveVideoIngestor:
    """Production Video Ingestion & Audio Slicing Engine for SieveRAG.

    Coordinates vlm-sieve motion gating and frame packing with sub-second
    mono audio extraction to deliver Phase 2 ready visual and audio artifacts.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        sieve_config: SieveConfig | None = None,
        **sieve_overrides: Any,
    ) -> None:
        """Initialize SieveVideoIngestor with vlm-sieve hyperparameters.

        Args:
            settings: Optional application settings instance.
            sieve_config: Optional pre-configured SieveConfig instance.
            **sieve_overrides: Optional keyword overrides for SieveConfig.
        """
        self.settings = settings or get_settings()
        self.ffmpeg_bin = get_ffmpeg_binary()

        # Phase 1 SieveConfig defaults
        default_params: dict[str, Any] = {
            "sample_fps": 10.0,
            "pre_roll_sec": 3.5,
            "merge_gap_sec": 6.0,
            "tile_max_dim": 640,
            "spatial_grid_overlay": True,
            "burn_timestamps": True,
            "suppress_disturbances": True,
            "motion_threshold": self.settings.SIEVE_MOTION_THRESHOLD,
            "tokens_per_image": self.settings.SIEVE_TOKENS_PER_IMAGE,
            "target_vlm": getattr(self.settings, "DEFAULT_TARGET_VLM", "gemini"),
        }

        if sieve_config is not None:
            merged = sieve_config.model_dump()
            merged.update(sieve_overrides)
            self.sieve_config = SieveConfig(**merged)
        else:
            merged = {**default_params, **sieve_overrides}
            self.sieve_config = SieveConfig(**merged)

        logger.info(
            "SieveVideoIngestor initialized (sample_fps=%.1f, pre_roll=%.1fs, merge_gap=%.1fs, tile_max_dim=%d)",
            self.sieve_config.sample_fps or 0.0,
            self.sieve_config.pre_roll_sec,
            self.sieve_config.merge_gap_sec,
            self.sieve_config.tile_max_dim,
        )

    def has_audio_stream(self, video_path: str | Path) -> bool:
        """Check whether the given media file contains at least one audio stream.

        Args:
            video_path: Path to the video file to inspect.

        Returns:
            True if an audio stream is detected; False otherwise.
        """
        vpath = Path(video_path).resolve()
        if not vpath.is_file():
            return False

        cmd = [self.ffmpeg_bin, "-hide_banner", "-i", str(vpath)]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            # Inspect stderr for Audio stream signatures (e.g. "Stream #0:1: Audio:")
            return bool(re.search(r"Stream\s+#\d+:\d+.*:\s+Audio:", proc.stderr, re.IGNORECASE))
        except Exception as exc:
            logger.warning("Error probing audio stream for %s: %s", vpath.name, exc)
            return False

    def get_video_metadata(self, video_path: str | Path) -> dict[str, Any]:
        """Extract baseline duration, dimensions, and frame counts from video.

        Args:
            video_path: Path to the video file.

        Returns:
            Dictionary containing duration_sec, fps, frame_count, width, height.
        """
        vpath = Path(video_path).resolve()
        if not vpath.is_file():
            raise FileNotFoundError(f"Source video not found: {video_path}")

        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file with OpenCV: {video_path}")

        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            if fps <= 0.0 or math.isnan(fps):
                fps = 30.0

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_sec = frame_count / fps if frame_count > 0 else 0.0

            # Fallback to imageio_ffmpeg if OpenCV reports 0 duration
            if duration_sec <= 0.0:
                try:
                    _, n_secs = imageio_ffmpeg.count_frames_and_secs(str(vpath))
                    duration_sec = float(n_secs)
                except Exception:
                    pass

            return {
                "duration_sec": max(0.01, round(duration_sec, 3)),
                "fps": round(fps, 2),
                "frame_count": max(1, frame_count),
                "width": width,
                "height": height,
            }
        finally:
            cap.release()

    def extract_episode_audio(
        self,
        video_path: str | Path,
        start_sec: float,
        end_sec: float,
        out_mp3_path: str | Path,
        pad_sec: float | None = None,
    ) -> str | None:
        """Extract a synchronized lightweight mono MP3 audio slice for an episode.

        Executes FFmpeg with:
        [-ss effective_start -to effective_end -vn -ac 1 -ar 16000 -b:a 32k]
        where padding (default 0.5s) is applied so speech/chimes at motion boundaries are preserved.

        Args:
            video_path: Path to the input video.
            start_sec: Start timestamp in seconds.
            end_sec: End timestamp in seconds.
            out_mp3_path: Destination path for the sliced .mp3 file.
            pad_sec: Optional padding in seconds around boundaries (defaults to settings.AUDIO_PAD_SEC).

        Returns:
            Absolute string path to the generated MP3 if audio is present and sliced,
            or None if the video has no audio stream or slicing could not proceed.
        """
        vpath = Path(video_path).resolve()
        out_path = Path(out_mp3_path).resolve()

        if not vpath.is_file():
            logger.warning("Video file does not exist: %s", vpath)
            return None

        # Verify whether input video contains an audio stream
        if not self.has_audio_stream(vpath):
            logger.debug("Video %s contains no audio stream; skipping audio slice", vpath.name)
            return None

        # Apply audio padding so audio chimes/speech at motion boundaries are not clipped
        effective_pad = self.settings.AUDIO_PAD_SEC if pad_sec is None else float(pad_sec)
        start = max(0.0, float(start_sec) - effective_pad)
        end = max(start + 0.1, float(end_sec) + effective_pad)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            str(vpath),
            "-vn",
            "-ac",
            str(self.settings.AUDIO_CHANNELS),
            "-ar",
            str(self.settings.AUDIO_SAMPLE_RATE),
            "-b:a",
            str(self.settings.AUDIO_BITRATE),
            str(out_path),
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=30,
            )

            if proc.returncode == 0 and out_path.is_file() and out_path.stat().st_size > 0:
                logger.debug("Successfully sliced audio -> %s (%d bytes)", out_path.name, out_path.stat().st_size)
                return str(out_path)

            logger.warning(
                "Audio slicing failed for [%.2fs - %.2fs]: %s",
                start,
                end,
                proc.stderr[-300:] if proc.stderr else "Empty output",
            )
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            return None

        except Exception as exc:
            logger.warning("Exception during audio slicing: %s", exc)
            if out_path.exists():
                out_path.unlink(missing_ok=True)
            return None

    def ingest_video(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        video_id: str | None = None,
    ) -> tuple[dict[str, Any], list[VisualEvent], list[DisturbanceRecord], TokenSavingsReport]:
        """Ingest a video, emitting storyboards, audio slices, and comprehensive receipt.

        Args:
            video_path: Path to the target source video.
            output_dir: Optional root output directory. Defaults to data/storage/<video_id>.
            video_id: Optional unique video identifier. Generates a UUID if None.

        Returns:
            A 4-tuple containing:
            1. receipt_dict: Serialized dictionary of VideoIngestionReceipt.
            2. events: List of emitted VisualEvent instances with storyboards.
            3. disturbances: List of suppressed DisturbanceRecord instances.
            4. savings_report: TokenSavingsReport quantifying frame & token compression.
        """
        vpath = Path(video_path).resolve()
        if not vpath.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        vid_id = video_id or f"vid_{uuid.uuid4().hex[:12]}"
        target_dir = Path(output_dir).resolve() if output_dir else (self.settings.STORAGE_DIR / vid_id).resolve()
        target_dir.mkdir(parents=True, exist_ok=True)

        storyboards_dir = target_dir / "storyboards"
        audio_dir = target_dir / "audio"
        storyboards_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Starting ingestion for %s (id=%s) -> %s",
            vpath.name,
            vid_id,
            target_dir,
        )

        # Measure end-to-end CPU processing latency
        t_start = time.perf_counter()

        # Extract source video metadata
        meta = self.get_video_metadata(vpath)
        duration_sec = meta["duration_sec"]

        # Run vlm-sieve motion gatekeeper, ROI cropper & storyboard packer
        sieve = VideoSieve(source=str(vpath), config=self.sieve_config)
        events: list[VisualEvent] = sieve.process_video_to_dir(output_dir=storyboards_dir)

        # Slice episode audio and construct IndexedEpisodeRecord instances
        episodes: list[IndexedEpisodeRecord] = []
        for ev in events:
            audio_filename = f"event_{ev.event_id:03d}.mp3"
            audio_out_path = audio_dir / audio_filename

            extracted_audio = self.extract_episode_audio(
                video_path=vpath,
                start_sec=ev.start_time_sec,
                end_sec=ev.end_time_sec,
                out_mp3_path=audio_out_path,
            )

            storyboard_filename = f"event_{ev.event_id:03d}.jpg"
            storyboard_full_path = storyboards_dir / storyboard_filename

            # Calculate relative path from storage directory or target_dir
            storyboard_rel = (
                storyboard_full_path.relative_to(target_dir).as_posix()
                if storyboard_full_path.is_relative_to(target_dir)
                else storyboard_filename
            )

            audio_rel = None
            if extracted_audio:
                audio_full_path = Path(extracted_audio)
                audio_rel = (
                    audio_full_path.relative_to(target_dir).as_posix()
                    if audio_full_path.is_relative_to(target_dir)
                    else audio_filename
                )

            record = IndexedEpisodeRecord.from_sieve_event(
                event=ev,
                video_id=vid_id,
                video_filename=vpath.name,
                storyboard_rel_path=storyboard_rel,
                audio_rel_path=audio_rel,
                sieve_input_tokens=self.sieve_config.tokens_per_image,
                sieve_output_tokens=0,
            )
            episodes.append(record)

        # Collect disturbances and savings reports from vlm-sieve
        disturbances: list[DisturbanceRecord] = sieve.get_disturbances()
        savings_report: TokenSavingsReport = sieve.get_savings_report()

        processing_wall_sec = round(time.perf_counter() - t_start, 3)

        # Calculate token metrics
        estimated_direct_video_tokens = int(round(duration_sec * self.settings.DIRECT_VIDEO_TOKENS_PER_SEC))
        actual_sieve_rag_tokens = len(events) * self.sieve_config.tokens_per_image

        if estimated_direct_video_tokens > 0:
            saved_tokens = max(0, estimated_direct_video_tokens - actual_sieve_rag_tokens)
            tokens_saved_pct = round((saved_tokens / float(estimated_direct_video_tokens)) * 100.0, 2)
        else:
            tokens_saved_pct = 0.0

        # Build VideoIngestionReceipt
        receipt = VideoIngestionReceipt(
            video_id=vid_id,
            video_filename=vpath.name,
            duration_sec=duration_sec,
            raw_frames_inspected=savings_report.raw_frames_inspected,
            sampled_frames_decoded=savings_report.sampled_frames_decoded,
            storyboards_emitted=len(events),
            disturbances_suppressed=len(disturbances),
            processing_wall_sec=processing_wall_sec,
            estimated_direct_video_tokens=estimated_direct_video_tokens,
            actual_sieve_rag_tokens=actual_sieve_rag_tokens,
            tokens_saved_pct=tokens_saved_pct,
            episodes=episodes,
            disturbances=[d.model_dump() for d in disturbances],
        )

        # Persist receipt to disk for auditability
        receipt_path = target_dir / "ingestion_receipt.json"
        with open(receipt_path, "w", encoding="utf-8") as f:
            json.dump(receipt.model_dump(), f, indent=2)

        logger.info(
            "Ingestion completed for %s in %.2fs: %d episodes, %d disturbances, %.1f%% tokens saved",
            vpath.name,
            processing_wall_sec,
            len(events),
            len(disturbances),
            tokens_saved_pct,
        )

        return (receipt.model_dump(), events, disturbances, savings_report)

    def ingest_video_to_receipt(
        self,
        video_path: str | Path,
        output_dir: str | Path | None = None,
        video_id: str | None = None,
    ) -> VideoIngestionReceipt:
        """Convenience method returning the typed VideoIngestionReceipt model directly.

        Args:
            video_path: Path to the target source video.
            output_dir: Optional root output directory.
            video_id: Optional unique video identifier.

        Returns:
            Validated VideoIngestionReceipt model instance.
        """
        receipt_dict, _, _, _ = self.ingest_video(
            video_path=video_path,
            output_dir=output_dir,
            video_id=video_id,
        )
        return VideoIngestionReceipt.model_validate(receipt_dict)

