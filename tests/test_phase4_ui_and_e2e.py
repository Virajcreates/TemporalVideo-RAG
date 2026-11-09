"""Comprehensive unit and end-to-end integration test suite for SieveRAG Phase 4:

Interactive Command Center UI Serving, Static Asset Mounting, Click-to-Seek Video Streaming,
Corpus Telemetry Aggregation, and Video Deletion Lifecycle.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
from fastapi.testclient import TestClient

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.config import Settings
from backend.app.extractor import StoryboardVLMExtractor
from backend.app.indexer import EpisodeEmbedder, SieveRAGVectorIndex
from backend.app.main import create_app
from backend.app.rag import SieveRAGSynthesizer
from backend.app.schemas import (
    EpisodeExtractionSchema,
    IndexedEpisodeRecord,
    TileAction,
)


def create_synthetic_mp4_file(
    output_path: Path,
    duration_frames: int = 30,
    fps: float = 10.0,
    width: int = 640,
    height: int = 360,
) -> Path:
    """Generate a physical .mp4 video file on disk with clear visual motion."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for i in range(duration_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if 8 <= i <= 22:
            # Moving bright yellow rectangle (simulating a forklift or person moving)
            x = 60 + i * 18
            cv2.rectangle(frame, (x, 80), (x + 80, 180), (0, 255, 255), -1)
        out.write(frame)
    out.release()
    return output_path


class TestPhase4UIAndE2E(unittest.TestCase):
    """Test suite covering Phase 4 UI, streaming, telemetry, and deletion."""

    def setUp(self) -> None:
        self.temp_storage = Path(tempfile.mkdtemp(prefix="sieverag_p4_storage_"))
        self.frontend_dir = WORKSPACE_ROOT / "frontend"

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_storage, ignore_errors=True)

    def test_root_serves_command_center_ui_and_static_assets(self) -> None:
        """Requirement 1: Verify GET / serves index.html and /static/... serves CSS and JS."""
        app = create_app(
            workspace_dir=self.temp_storage,
            frontend_dir=self.frontend_dir,
        )
        client = TestClient(app)

        # 1. Test GET / serves index.html
        res_root = client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertIn("text/html", res_root.headers.get("content-type", ""))
        body_text = res_root.text
        self.assertIn("SieveRAG", body_text)
        self.assertIn("main-video-player", body_text)
        self.assertIn("telemetry-bar", body_text)
        self.assertIn("timeline-track", body_text)
        self.assertIn("rag-query-input", body_text)

        # 2. Test GET /static/styles.css serves valid CSS
        res_css = client.get("/static/styles.css")
        self.assertEqual(res_css.status_code, 200)
        self.assertIn("text/css", res_css.headers.get("content-type", ""))
        self.assertIn("--bg-main: #0b0f17;", res_css.text)
        self.assertIn(".command-center-layout", res_css.text)

        # 3. Test GET /static/app.js serves valid JavaScript
        res_js = client.get("/static/app.js")
        self.assertEqual(res_js.status_code, 200)
        self.assertIn("javascript", res_js.headers.get("content-type", ""))
        self.assertIn("SieveRAG Command Center", res_js.text)
        self.assertIn("fetchStats", res_js.text)
        self.assertIn("loadVideo", res_js.text)

    def test_ingest_local_video_path_and_video_url_streaming(self) -> None:
        """Requirement 2: Ingest via local_video_path, verify video_url and /media streaming."""
        test_video_path = self.temp_storage / "incoming" / "surveillance_test.mp4"
        create_synthetic_mp4_file(test_video_path, duration_frames=30, fps=10.0)

        mock_extractor = MagicMock(spec=StoryboardVLMExtractor)
        mock_extractor.client = None
        mock_extractor.api_key = ""

        app = create_app(
            workspace_dir=self.temp_storage,
            frontend_dir=self.frontend_dir,
            extractor=mock_extractor,
        )
        client = TestClient(app)

        # Ingest using local_video_path form data
        res_ingest = client.post(
            "/api/videos/ingest",
            data={
                "local_video_path": str(test_video_path),
                "target_vlm": "gemini",
                "run_vlm_extraction": "true",
                "index_episodes": "true",
            },
        )
        self.assertEqual(res_ingest.status_code, 200)
        data = res_ingest.json()

        video_id = data["video_id"]
        self.assertTrue(video_id.startswith("vid_"))
        self.assertIn("video_url", data)
        self.assertTrue(data["video_url"].startswith("/media/"))
        self.assertIn("surveillance_test.mp4", data["video_url"])

        # Stream video file directly from the resolved video_url
        res_stream = client.get(data["video_url"])
        self.assertEqual(res_stream.status_code, 200)
        self.assertGreater(len(res_stream.content), 100)

        # Query GET /api/videos/{video_id}
        res_details = client.get(f"/api/videos/{video_id}")
        self.assertEqual(res_details.status_code, 200)
        details = res_details.json()
        self.assertEqual(details["video_url"], data["video_url"])
        self.assertGreaterEqual(len(details["episodes"]), 1)

        first_ep = details["episodes"][0]
        self.assertIn("storyboard_url", first_ep)
        self.assertTrue(first_ep["storyboard_url"].startswith("/media/"))

        # Verify storyboard is accessible via HTTP
        res_sb = client.get(first_ep["storyboard_url"])
        self.assertEqual(res_sb.status_code, 200)
        self.assertIn("image/jpeg", res_sb.headers.get("content-type", ""))

    def test_global_stats_endpoint(self) -> None:
        """Requirement 3: Verify GET /api/stats calculates corpus telemetry accurately."""
        test_video_path = self.temp_storage / "incoming" / "cam2_motion.mp4"
        create_synthetic_mp4_file(test_video_path, duration_frames=30, fps=10.0)

        mock_extractor = MagicMock(spec=StoryboardVLMExtractor)
        mock_extractor.client = None
        mock_extractor.api_key = ""

        app = create_app(
            workspace_dir=self.temp_storage,
            frontend_dir=self.frontend_dir,
            extractor=mock_extractor,
        )
        client = TestClient(app)

        # Before ingestion: all stats are zero
        res_empty = client.get("/api/stats")
        self.assertEqual(res_empty.status_code, 200)
        stats_empty = res_empty.json()
        self.assertEqual(stats_empty["total_videos"], 0)
        self.assertEqual(stats_empty["indexed_episodes_count"], 0)

        # Ingest video
        client.post(
            "/api/videos/ingest",
            data={
                "local_video_path": str(test_video_path),
                "target_vlm": "gemini",
                "run_vlm_extraction": "true",
                "index_episodes": "true",
            },
        )

        # After ingestion: verify telemetry metrics
        res_stats = client.get("/api/stats")
        self.assertEqual(res_stats.status_code, 200)
        stats = res_stats.json()

        self.assertEqual(stats["total_videos"], 1)
        self.assertGreater(stats["total_duration_sec"], 0.0)
        self.assertGreater(stats["total_raw_frames_inspected"], 0)
        self.assertGreaterEqual(stats["total_storyboards_emitted"], 1)
        self.assertGreaterEqual(stats["overall_tokens_saved_pct"], 0.0)
        self.assertIn("total_qwen_vl_patch_tokens", stats)
        self.assertIn("total_qwen7b_kv_cache_mb", stats)
        self.assertGreaterEqual(stats["indexed_episodes_count"], 1)
        self.assertEqual(stats["vector_backend"], "local")

    def test_delete_video_removes_storage_and_vectors(self) -> None:
        """Requirement 4: Verify DELETE /api/videos/{video_id} purges storage and vectors."""
        test_video_path = self.temp_storage / "incoming" / "cam_delete.mp4"
        create_synthetic_mp4_file(test_video_path, duration_frames=30, fps=10.0)

        mock_extractor = MagicMock(spec=StoryboardVLMExtractor)
        mock_extractor.client = None
        mock_extractor.api_key = ""

        app = create_app(
            workspace_dir=self.temp_storage,
            frontend_dir=self.frontend_dir,
            extractor=mock_extractor,
        )
        client = TestClient(app)

        # Ingest video
        res_ingest = client.post(
            "/api/videos/ingest",
            data={
                "local_video_path": str(test_video_path),
                "target_vlm": "gemini",
                "run_vlm_extraction": "true",
                "index_episodes": "true",
            },
        )
        video_id = res_ingest.json()["video_id"]
        video_dir = self.temp_storage / video_id
        self.assertTrue(video_dir.exists())

        # Verify it shows up in stats and list
        self.assertEqual(client.get("/api/stats").json()["total_videos"], 1)
        self.assertEqual(len(client.get("/api/videos").json()), 1)

        # Delete the video
        res_del = client.delete(f"/api/videos/{video_id}")
        self.assertEqual(res_del.status_code, 200)
        del_data = res_del.json()
        self.assertTrue(del_data["deleted"])
        self.assertEqual(del_data["video_id"], video_id)
        self.assertGreaterEqual(del_data["removed_vectors"], 1)

        # Storage directory should be purged
        self.assertFalse(video_dir.exists())

        # GET /api/videos/{video_id} must return 404
        res_get = client.get(f"/api/videos/{video_id}")
        self.assertEqual(res_get.status_code, 404)

        # Corpus stats must be updated back to 0
        stats_after = client.get("/api/stats").json()
        self.assertEqual(stats_after["total_videos"], 0)
        self.assertEqual(stats_after["indexed_episodes_count"], 0)
        self.assertEqual(len(client.get("/api/videos").json()), 0)


if __name__ == "__main__":
    unittest.main()

