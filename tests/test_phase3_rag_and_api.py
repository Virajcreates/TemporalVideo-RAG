"""Comprehensive unit and integration test suite for SieveRAG Phase 3:

Temporal Video RAG Answer Synthesizer, FastAPI Endpoints, and End-to-End API Workflows.
Uses fastapi.testclient.TestClient with mocked VLM/synthesizer clients and offline modes.
"""

from __future__ import annotations

import io
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
from PIL import Image

# Ensure workspace root is on sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.config import Settings
from backend.app.extractor import StoryboardVLMExtractor
from backend.app.indexer import EpisodeEmbedder, SieveRAGVectorIndex
from backend.app.main import create_app
from backend.app.rag import SieveRAGSynthesizer, build_rag_prompt
from backend.app.schemas import (
    AskRequest,
    EpisodeExtractionSchema,
    IndexedEpisodeRecord,
    RAGAnswerSchema,
    TemporalCitation,
    TileAction,
)


def create_synthetic_mp4_bytes(
    duration_frames: int = 25,
    fps: float = 10.0,
    width: int = 640,
    height: int = 360,
) -> bytes:
    """Generate in-memory bytes of a valid H.264 / mp4v video with distinct motion."""
    tmp_path = Path(tempfile.mktemp(suffix=".mp4"))
    out = cv2.VideoWriter(
        str(tmp_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for i in range(duration_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if 8 <= i <= 20:
            # Moving bright rectangle across frame
            x = 80 + i * 20
            cv2.rectangle(frame, (x, 100), (x + 80, 200), (0, 255, 255), -1)
        out.write(frame)
    out.release()
    data = tmp_path.read_bytes()
    tmp_path.unlink(missing_ok=True)
    return data


class TestPhase3RAGAndAPI(unittest.TestCase):
    """Test suite for Phase 3 FastAPI API and RAG Answer Synthesizer."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sieverag_p3_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_health_endpoint(self) -> None:
        """Requirement 1: Verify GET /api/health returns 200, vlmsieve_version == '0.3.0',

        and correct storage/backend configuration metadata.
        """
        app = create_app(workspace_dir=self.temp_dir)
        client = TestClient(app)

        res = client.get("/api/health")
        self.assertEqual(res.status_code, 200)

        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["vlmsieve_version"], "0.3.0")
        self.assertEqual(data["target_vlm_default"], "gemini")
        self.assertIn(data["vector_backend"], ("local", "pinecone"))
        self.assertEqual(data["storage_dir"], str(self.temp_dir.resolve()))
        self.assertIsInstance(data["indexed_episodes_count"], int)

    def test_ingest_video_upload_and_list_videos(self) -> None:
        """Requirement 2: Upload synthetic motion video via POST /api/videos/ingest

        with target_vlm='qwen-vl', verify 200 OK, 28px-aligned storyboard, receipt persistence,
        GET /api/videos, GET /api/videos/{video_id}, and static serving via GET /media/...
        """
        video_bytes = create_synthetic_mp4_bytes(duration_frames=25)

        # Mock VLM extractor so extraction succeeds deterministically
        mock_extraction = EpisodeExtractionSchema(
            episode_title="Moving Cargo Box",
            narrative_summary="A bright cargo box traverses from B1 to C3 across the frame.",
            actors=["automated cart"],
            objects_detected=["yellow cargo box"],
            audio_transcript_and_cues="Silent operation.",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=0.0,
                    absolute_time_sec=0.8,
                    grid_cells=["B1"],
                    action_description="Cargo box enters view.",
                ),
                TileAction(
                    tile_number=2,
                    relative_time_sec=1.0,
                    absolute_time_sec=1.8,
                    grid_cells=["C3"],
                    action_description="Cargo box reaches waypoint.",
                ),
            ],
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_extraction
        mock_response.text = mock_extraction.model_dump_json()
        mock_response.usage_metadata.prompt_token_count = 1120
        mock_response.usage_metadata.candidates_token_count = 140

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        extractor = StoryboardVLMExtractor(client=mock_client)
        app = create_app(workspace_dir=self.temp_dir, extractor=extractor)
        client = TestClient(app)

        # Upload video with target_vlm="qwen-vl"
        upload_resp = client.post(
            "/api/videos/ingest",
            files={"file": ("warehouse_cctv.mp4", video_bytes, "video/mp4")},
            data={
                "target_vlm": "qwen-vl",
                "run_vlm_extraction": "true",
                "index_episodes": "true",
            },
        )
        self.assertEqual(upload_resp.status_code, 200, upload_resp.text)
        ingest_data = upload_resp.json()

        video_id = ingest_data["video_id"]
        self.assertEqual(ingest_data["video_filename"], "warehouse_cctv.mp4")
        self.assertGreater(len(ingest_data["episodes"]), 0)
        self.assertGreaterEqual(ingest_data["upserted_vectors"], 1)

        # Check receipt persistence on disk
        receipt_on_disk = self.temp_dir / video_id / "ingestion_receipt.json"
        self.assertTrue(receipt_on_disk.is_file())

        # Verify Qwen-VL 28px patch alignment on storyboard JPEG
        ep0 = ingest_data["episodes"][0]
        sb_rel = ep0["storyboard_rel_path"]
        sb_disk_path = self.temp_dir / video_id / sb_rel
        if not sb_disk_path.is_file():
            # If path already contains video_id
            sb_disk_path = self.temp_dir / sb_rel

        self.assertTrue(sb_disk_path.is_file(), f"Storyboard not found at {sb_disk_path}")
        with Image.open(sb_disk_path) as im:
            sb_w, sb_h = im.size
            self.assertEqual(
                sb_w % 28,
                0,
                f"Storyboard width ({sb_w}) must be divisible by 28 for Qwen-VL patch alignment!",
            )
            self.assertEqual(
                sb_h % 28,
                0,
                f"Storyboard height ({sb_h}) must be divisible by 28 for Qwen-VL patch alignment!",
            )

        # Verify GET /api/videos lists the ingested video
        list_resp = client.get("/api/videos")
        self.assertEqual(list_resp.status_code, 200)
        video_list = list_resp.json()
        matching = [v for v in video_list if v["video_id"] == video_id]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["video_filename"], "warehouse_cctv.mp4")

        # Verify GET /api/videos/{video_id}
        details_resp = client.get(f"/api/videos/{video_id}")
        self.assertEqual(details_resp.status_code, 200)
        details = details_resp.json()
        self.assertEqual(details["video_id"], video_id)
        self.assertTrue(details["episodes"][0]["storyboard_url"].startswith("/media/"))

        # Verify StaticFiles serving: GET {storyboard_url}
        sb_url = details["episodes"][0]["storyboard_url"]
        media_resp = client.get(sb_url)
        self.assertEqual(media_resp.status_code, 200)
        self.assertEqual(media_resp.headers.get("content-type"), "image/jpeg")
        self.assertGreater(len(media_resp.content), 0)

    def test_search_and_ask_endpoints_with_citations(self) -> None:
        """Requirement 3: Test POST /api/search and POST /api/ask with citations,

        verifying exact timestamp/tile/grid citations, /media/... links, and anomaly_only filtering.
        """
        # Create dummy storyboard image files in storage
        sb_dir1 = self.temp_dir / "vid_gate_01" / "storyboards"
        sb_dir2 = self.temp_dir / "vid_gate_02" / "storyboards"
        sb_dir1.mkdir(parents=True, exist_ok=True)
        sb_dir2.mkdir(parents=True, exist_ok=True)

        (sb_dir1 / "event_001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01")
        (sb_dir2 / "event_001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01")

        # Seed vector index with 2 distinct episodes
        embedder = EpisodeEmbedder(use_mock_embeddings=True)
        index = SieveRAGVectorIndex(
            embedder=embedder,
            local_index_path=self.temp_dir / "vector_index.json",
        )

        ep1 = IndexedEpisodeRecord(
            video_id="vid_gate_01",
            video_filename="perimeter_cam.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=10.0,
            end_time_sec=16.0,
            duration_sec=6.0,
            storyboard_rel_path="vid_gate_01/storyboards/event_001.jpg",
            episode_title="Fence Breach with Bolt Cutters",
            narrative_summary="A masked suspect cuts through the chain-link perimeter fence using heavy bolt cutters.",
            actors=["masked suspect"],
            objects_detected=["bolt cutters", "chain-link fence", "dark hoodie"],
            audio_transcript_and_cues="Metallic snapping sound followed by fence rattling.",
            anomaly_or_security_flag=True,
            severity="high",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=0.0,
                    absolute_time_sec=10.0,
                    grid_cells=["A1", "B1"],
                    action_description="Suspect approaches fence line.",
                ),
                TileAction(
                    tile_number=2,
                    relative_time_sec=2.1,
                    absolute_time_sec=12.1,
                    grid_cells=["B2", "C3"],
                    action_description="Suspect cuts fence link with bolt cutters.",
                ),
            ],
        )

        ep2 = IndexedEpisodeRecord(
            video_id="vid_gate_02",
            video_filename="front_gate.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=0.0,
            end_time_sec=5.0,
            duration_sec=5.0,
            storyboard_rel_path="vid_gate_02/storyboards/event_001.jpg",
            episode_title="Postal Delivery Vehicle",
            narrative_summary="Postal delivery van drives through the front gate during regular operating hours.",
            actors=["postal driver"],
            objects_detected=["delivery van", "gate"],
            audio_transcript_and_cues="Engine idle and gate motor humming.",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=1.0,
                    absolute_time_sec=1.0,
                    grid_cells=["C2"],
                    action_description="Van passes gate scanner.",
                )
            ],
        )

        index.upsert_episodes([ep1, ep2])

        # Mock Gemini Synthesizer client
        mock_rag_answer = RAGAnswerSchema(
            answer=(
                "A masked suspect was observed cutting the perimeter chain-link fence using bolt cutters "
                "at Tile #2 (+2.10s / 12.10s in spatial grid cells B2, C3)."
            ),
            confidence="high",
            security_alert_triggered=True,
            citations=[
                TemporalCitation(
                    video_id="vid_gate_01",
                    episode_index=1,
                    start_time_sec=10.0,
                    end_time_sec=16.0,
                    tile_number=2,
                    timestamp_sec=12.1,
                    grid_cells=["B2", "C3"],
                    evidence="Suspect cuts fence link with bolt cutters in grid B2,C3.",
                    storyboard_url="/media/vid_gate_01/storyboards/event_001.jpg",
                )
            ],
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_rag_answer
        mock_response.text = mock_rag_answer.model_dump_json()
        mock_response.usage_metadata.prompt_token_count = 850
        mock_response.usage_metadata.candidates_token_count = 110

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        synthesizer = SieveRAGSynthesizer(
            index=index,
            workspace_dir=self.temp_dir,
            client=mock_client,
        )

        app = create_app(
            workspace_dir=self.temp_dir,
            index=index,
            synthesizer=synthesizer,
        )
        client = TestClient(app)

        # 1. Test POST /api/search
        search_res = client.post(
            "/api/search",
            json={"query": "fence breach bolt cutters", "top_k": 2},
        )
        self.assertEqual(search_res.status_code, 200)
        s_data = search_res.json()
        self.assertEqual(s_data["count"], 2)
        top_hit = s_data["results"][0]
        self.assertEqual(top_hit["episode_title"], "Fence Breach with Bolt Cutters")
        self.assertTrue(top_hit["storyboard_url"].startswith("/media/"))

        # Test anomaly_only filter on search
        anomaly_search_res = client.post(
            "/api/search",
            json={"query": "gate activity", "top_k": 5, "anomaly_only": True},
        )
        self.assertEqual(anomaly_search_res.status_code, 200)
        a_data = anomaly_search_res.json()
        self.assertEqual(a_data["count"], 1)
        self.assertEqual(a_data["results"][0]["video_id"], "vid_gate_01")

        # 2. Test POST /api/ask
        ask_res = client.post(
            "/api/ask",
            json={
                "question": "Did anyone tamper with the perimeter fence?",
                "top_k": 2,
                "include_storyboards": True,
            },
        )
        self.assertEqual(ask_res.status_code, 200)
        ask_data = ask_res.json()

        self.assertTrue(ask_data["security_alert_triggered"])
        self.assertEqual(ask_data["confidence"], "high")
        self.assertIn("bolt cutters", ask_data["answer"])
        self.assertEqual(len(ask_data["citations"]), 1)

        citation = ask_data["citations"][0]
        self.assertEqual(citation["video_id"], "vid_gate_01")
        self.assertEqual(citation["tile_number"], 2)
        self.assertAlmostEqual(citation["timestamp_sec"], 12.1)
        self.assertEqual(citation["grid_cells"], ["B2", "C3"])
        self.assertTrue(citation["storyboard_url"].startswith("/media/"))

    def test_offline_fallback_mode_without_api_key(self) -> None:
        """Requirement 4: Verify that when GEMINI_API_KEY='' and no mock client is injected,

        POST /api/videos/ingest and POST /api/ask still succeed gracefully using the
        deterministic offline fallback without throwing 500 errors.
        """
        video_bytes = create_synthetic_mp4_bytes(duration_frames=25)

        # Force empty API keys in settings
        offline_settings = Settings(
            GEMINI_API_KEY="",
            PINECONE_API_KEY="",
        )

        app = create_app(
            workspace_dir=self.temp_dir,
            settings=offline_settings,
        )
        client = TestClient(app)

        # Ingestion in offline mode
        ingest_res = client.post(
            "/api/videos/ingest",
            files={"file": ("offline_feed.mp4", video_bytes, "video/mp4")},
            data={"run_vlm_extraction": "true", "index_episodes": "true"},
        )
        self.assertEqual(ingest_res.status_code, 200, ingest_res.text)
        ingest_json = ingest_res.json()

        # Offline fallback extraction should populate baseline title and summary
        self.assertGreater(len(ingest_json["episodes"]), 0)
        first_ep = ingest_json["episodes"][0]
        self.assertIn("Motion Episode", first_ep["episode_title"])
        self.assertIn("Detected visual motion activity", first_ep["narrative_summary"])

        # Ask in offline mode
        ask_res = client.post(
            "/api/ask",
            json={"question": "What motion occurred in the scene?", "top_k": 2},
        )
        self.assertEqual(ask_res.status_code, 200, ask_res.text)
        ask_json = ask_res.json()

        self.assertIn("Based on indexed video", ask_json["answer"])
        self.assertIn(ask_json["confidence"], ("medium", "high"))
        self.assertGreater(len(ask_json["citations"]), 0)
        self.assertTrue(ask_json["citations"][0]["storyboard_url"].startswith("/media/"))


if __name__ == "__main__":
    unittest.main()

