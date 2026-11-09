"""Comprehensive test suite for SieveRAG ingestion, audio slicing, and schema validation."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure root workspace is on python sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.config import Settings, get_settings
from backend.app.ingestor import SieveVideoIngestor, get_ffmpeg_binary
from backend.app.schemas import (
    EpisodeExtractionSchema,
    IndexedEpisodeRecord,
    TileAction,
    VideoIngestionReceipt,
)
from vlmsieve import SieveConfig
from vlmsieve.models import DisturbanceRecord, TokenSavingsReport, VisualEvent


class TestSieveRAGSchemas(unittest.TestCase):
    """Test validation and serialization of Pydantic schemas."""

    def test_tile_action_valid(self) -> None:
        action = TileAction(
            tile_number=1,
            relative_time_sec=0.0,
            absolute_time_sec=12.4,
            grid_cells=["B2", "C3"],
            action_description="Subject approaches entrance.",
        )
        self.assertEqual(action.tile_number, 1)
        self.assertEqual(action.grid_cells, ["B2", "C3"])

    def test_tile_action_invalid_tile_number(self) -> None:
        with self.assertRaises(Exception):
            TileAction(
                tile_number=7,  # Must be 1..6
                relative_time_sec=0.0,
                absolute_time_sec=0.0,
                grid_cells=[],
                action_description="Invalid tile number",
            )

    def test_episode_extraction_schema_defaults(self) -> None:
        schema = EpisodeExtractionSchema(
            episode_title="Person Entering Hallway",
            narrative_summary="A person enters the hallway from doorway B1 and walks toward C3.",
            actors=["person_1"],
            objects_detected=["backpack", "door"],
            audio_transcript_and_cues="Footsteps echoing in hallway",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=0.0,
                    absolute_time_sec=5.0,
                    grid_cells=["B1"],
                    action_description="Person enters frame",
                )
            ],
        )
        data = schema.model_dump()
        self.assertEqual(data["severity"], "low")
        self.assertEqual(len(data["tile_actions"]), 1)

    def test_indexed_episode_record_combines_metadata(self) -> None:
        record = IndexedEpisodeRecord(
            video_id="vid_test_123",
            video_filename="test.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=1.5,
            end_time_sec=4.5,
            duration_sec=3.0,
            keyframe_times_sec=[1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
            roi_bbox={"x": 50, "y": 50, "width": 300, "height": 200},
            storyboard_rel_path="storyboards/event_001.jpg",
            audio_rel_path="audio/event_001.mp3",
            sieve_input_tokens=1120,
            sieve_output_tokens=0,
            episode_title="Test Episode",
            narrative_summary="Test summary",
            actors=["subject"],
            objects_detected=[],
            audio_transcript_and_cues="",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[],
        )
        self.assertEqual(record.video_id, "vid_test_123")
        self.assertEqual(record.duration_sec, 3.0)
        self.assertEqual(record.sieve_input_tokens, 1120)

        # Test extraction extraction
        extraction = record.to_extraction_schema()
        self.assertEqual(extraction.episode_title, "Test Episode")

        # Test updating with extraction
        updated_extraction = EpisodeExtractionSchema(
            episode_title="Updated Title",
            narrative_summary="Updated summary",
            actors=["updated_actor"],
            objects_detected=["box"],
            audio_transcript_and_cues="Quiet",
            anomaly_or_security_flag=True,
            severity="high",
            tile_actions=[],
        )
        record.apply_extraction(updated_extraction, output_tokens=350)
        self.assertEqual(record.episode_title, "Updated Title")
        self.assertTrue(record.anomaly_or_security_flag)
        self.assertEqual(record.severity, "high")
        self.assertEqual(record.sieve_output_tokens, 350)


class TestSieveVideoIngestor(unittest.TestCase):
    """Integration tests for SieveVideoIngestor and audio slicing."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="sieverag_test_"))
        cls.ffmpeg = get_ffmpeg_binary()

        # 1. Create a synthetic video WITHOUT audio (3 seconds, 10 fps)
        cls.no_audio_video = cls.temp_dir / "synthetic_no_audio.mp4"
        h, w = 360, 640
        out = cv2.VideoWriter(
            str(cls.no_audio_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (w, h),
        )
        # First 1 second static, 1.5 seconds motion, 0.5s static
        for i in range(30):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if 10 <= i <= 25:
                cv2.circle(frame, (100 + i * 15, 180), 30, (255, 255, 255), -1)
            out.write(frame)
        out.release()

        # 2. Create a synthetic video WITH audio by merging synthetic_no_audio with sine audio
        cls.with_audio_video = cls.temp_dir / "synthetic_with_audio.mp4"
        cmd = [
            cls.ffmpeg,
            "-y",
            "-i",
            str(cls.no_audio_video),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=800:duration=3",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-shortest",
            str(cls.with_audio_video),
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_ffmpeg_binary_resolved(self) -> None:
        self.assertTrue(Path(self.ffmpeg).is_file())

    def test_audio_stream_detection(self) -> None:
        ingestor = SieveVideoIngestor()
        self.assertFalse(ingestor.has_audio_stream(self.no_audio_video))
        self.assertTrue(ingestor.has_audio_stream(self.with_audio_video))

    def test_extract_audio_graceful_on_no_audio_video(self) -> None:
        ingestor = SieveVideoIngestor()
        out_mp3 = self.temp_dir / "should_not_exist.mp3"
        result = ingestor.extract_episode_audio(
            video_path=self.no_audio_video,
            start_sec=0.5,
            end_sec=2.0,
            out_mp3_path=out_mp3,
        )
        self.assertIsNone(result)
        self.assertFalse(out_mp3.exists())

    def test_extract_audio_success_on_audio_video(self) -> None:
        ingestor = SieveVideoIngestor()
        out_mp3 = self.temp_dir / "episode_audio.mp3"
        result = ingestor.extract_episode_audio(
            video_path=self.with_audio_video,
            start_sec=1.0,
            end_sec=3.0,
            out_mp3_path=out_mp3,
        )
        self.assertIsNotNone(result)
        self.assertTrue(out_mp3.is_file())
        self.assertGreater(out_mp3.stat().st_size, 0)

    def test_ingest_video_complete_pipeline(self) -> None:
        output_run_dir = self.temp_dir / "ingest_run"
        ingestor = SieveVideoIngestor(
            sample_fps=10.0,
            pre_roll_sec=1.0,
            merge_gap_sec=2.0,
            tile_max_dim=640,
            spatial_grid_overlay=True,
            burn_timestamps=True,
            suppress_disturbances=True,
        )

        receipt_dict, events, disturbances, savings = ingestor.ingest_video(
            video_path=self.no_audio_video,
            output_dir=output_run_dir,
            video_id="test_vid_pipeline",
        )

        # 1. Check returned tuple types
        self.assertIsInstance(receipt_dict, dict)
        self.assertIsInstance(events, list)
        self.assertIsInstance(disturbances, list)
        self.assertIsInstance(savings, TokenSavingsReport)

        # 2. Check receipt fields
        receipt = VideoIngestionReceipt.model_validate(receipt_dict)
        self.assertEqual(receipt.video_id, "test_vid_pipeline")
        self.assertEqual(receipt.video_filename, "synthetic_no_audio.mp4")
        self.assertGreater(receipt.duration_sec, 0.0)
        self.assertGreaterEqual(receipt.raw_frames_inspected, 30)
        self.assertGreaterEqual(receipt.storyboards_emitted, 1)
        self.assertGreaterEqual(receipt.processing_wall_sec, 0.0)
        self.assertGreater(receipt.estimated_direct_video_tokens, 0)
        self.assertGreaterEqual(receipt.actual_sieve_rag_tokens, 1120)
        self.assertGreaterEqual(receipt.tokens_saved_pct, 0.0)

        # 3. Check emitted events and files
        self.assertEqual(len(receipt.episodes), len(events))
        for ep in receipt.episodes:
            self.assertEqual(ep.video_id, "test_vid_pipeline")
            self.assertEqual(ep.sieve_input_tokens, 1120)
            storyboard_path = output_run_dir / ep.storyboard_rel_path
            self.assertTrue(storyboard_path.is_file(), f"Storyboard not found: {storyboard_path}")
            # Video had no audio, so audio_rel_path must be None
            self.assertIsNone(ep.audio_rel_path)

        # 4. Check ingestion_receipt.json persisted on disk
        receipt_file = output_run_dir / "ingestion_receipt.json"
        self.assertTrue(receipt_file.is_file())
        with open(receipt_file, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        self.assertEqual(persisted["video_id"], "test_vid_pipeline")


    def test_ingest_video_with_audio_pipeline(self) -> None:
        output_run_dir = self.temp_dir / "ingest_run_with_audio"
        ingestor = SieveVideoIngestor(
            sample_fps=10.0,
            pre_roll_sec=1.0,
            merge_gap_sec=2.0,
            tile_max_dim=640,
            spatial_grid_overlay=True,
            burn_timestamps=True,
            suppress_disturbances=True,
        )

        receipt_dict, events, disturbances, savings = ingestor.ingest_video(
            video_path=self.with_audio_video,
            output_dir=output_run_dir,
            video_id="test_vid_audio_pipeline",
        )

        receipt = VideoIngestionReceipt.model_validate(receipt_dict)
        self.assertEqual(receipt.video_id, "test_vid_audio_pipeline")
        self.assertGreater(len(receipt.episodes), 0)
        for ep in receipt.episodes:
            self.assertIsNotNone(ep.audio_rel_path)
            audio_path = output_run_dir / ep.audio_rel_path
            self.assertTrue(audio_path.is_file())
            self.assertGreater(audio_path.stat().st_size, 0)


    def test_disturbance_suppression(self) -> None:
        # Create a video with global full-frame exposure flashing (disturbance)
        flash_video = self.temp_dir / "synthetic_flash.mp4"
        h, w = 360, 640
        out = cv2.VideoWriter(
            str(flash_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (w, h),
        )
        for i in range(25):
            # Alternate black and white full frames (global camera/exposure disturbance)
            color = 255 if (i % 2 == 0) else 0
            frame = np.full((h, w, 3), color, dtype=np.uint8)
            out.write(frame)
        out.release()

        ingestor = SieveVideoIngestor(
            sample_fps=10.0,
            suppress_disturbances=True,
        )
        output_dir = self.temp_dir / "ingest_flash"
        receipt_dict, events, disturbances, savings = ingestor.ingest_video(
            video_path=flash_video,
            output_dir=output_dir,
            video_id="test_flash_suppression",
        )
        receipt = VideoIngestionReceipt.model_validate(receipt_dict)
        self.assertEqual(receipt.video_id, "test_flash_suppression")
        # Global flashes should either be suppressed as disturbances or produce 0 localized storyboards
        self.assertEqual(len(receipt.episodes), 0)


if __name__ == "__main__":
    unittest.main()
