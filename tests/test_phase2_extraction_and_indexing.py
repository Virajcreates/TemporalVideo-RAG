"""Comprehensive unit and integration test suite for SieveRAG Phase 2:

Multimodal VLM Extraction & Vector Indexing Engine.
Tests all requirements with zero live network API dependencies using mocks and local stores.
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

# Ensure root workspace is on python sys.path
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.app.config import Settings, get_settings
from backend.app.extractor import StoryboardVLMExtractor, build_extraction_prompt
from backend.app.indexer import EpisodeEmbedder, SieveRAGVectorIndex
from backend.app.ingestor import SieveVideoIngestor
from backend.app.schemas import (
    EpisodeExtractionSchema,
    IndexedEpisodeRecord,
    TileAction,
    VideoIngestionReceipt,
)


class TestPhase2ExtractionAndIndexing(unittest.TestCase):
    """Test suite covering Phase 2 VLM extraction, vector indexing, and end-to-end pipeline."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sieverag_p2_test_"))

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_v030_schema_fields_and_metadata_serialization(self) -> None:
        """Requirement 1: Verify grid_used, to_embedding_text(), and to_vector_metadata()

        Ensures zero None values exist in Pinecone metadata even when audio_rel_path=None.
        """
        record = IndexedEpisodeRecord(
            video_id="vid_test_p2",
            video_filename="front_door.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=10.0,
            end_time_sec=15.0,
            duration_sec=5.0,
            keyframe_times_sec=[10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            roi_bbox={"x": 10, "y": 20, "width": 300, "height": 200},
            storyboard_rel_path="storyboards/event_001.jpg",
            audio_rel_path=None,  # Crucial: test None audio handling
            sieve_input_tokens=1120,
            sieve_output_tokens=150,
            grid_used=(2, 3),
            qwen_vl_patch_tokens=180,
            qwen7b_kv_cache_mb=1.45,
            episode_title="Delivery Attempt",
            narrative_summary="A courier approaches the doorstep holding a brown carton.",
            actors=["courier"],
            objects_detected=["brown carton", "cap"],
            audio_transcript_and_cues="No audio stream or silent episode.",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=0.0,
                    absolute_time_sec=10.0,
                    grid_cells=["A2", "B2"],
                    action_description="Courier approaches walkway.",
                ),
                TileAction(
                    tile_number=2,
                    relative_time_sec=1.0,
                    absolute_time_sec=11.0,
                    grid_cells=["B2", "C3"],
                    action_description="Courier places parcel near front door.",
                ),
            ],
        )

        # 1. Verify v0.3.0 fields
        self.assertEqual(record.grid_used, (2, 3))
        self.assertEqual(record.qwen_vl_patch_tokens, 180)
        self.assertAlmostEqual(record.qwen7b_kv_cache_mb, 1.45)

        # 2. Verify to_embedding_text()
        emb_text = record.to_embedding_text()
        self.assertIn("Title: Delivery Attempt", emb_text)
        self.assertIn("Summary: A courier approaches", emb_text)
        self.assertIn("Actors: courier", emb_text)
        self.assertIn("Objects: brown carton, cap", emb_text)
        self.assertIn("Severity: low (Anomaly: False)", emb_text)
        self.assertIn("[Tile #1 +0.00s @ A2,B2] Courier approaches walkway.", emb_text)
        self.assertIn("[Tile #2 +1.00s @ B2,C3] Courier places parcel near front door.", emb_text)

        # 3. Verify to_vector_metadata() and check for ZERO None values (Pinecone requirement)
        metadata = record.to_vector_metadata()
        for key, val in metadata.items():
            self.assertIsNotNone(val, f"Metadata key '{key}' contains None, which Pinecone will reject!")
            # Pinecone accepts str, int, float, bool, and list[str]
            self.assertTrue(
                isinstance(val, (str, int, float, bool, list)),
                f"Metadata key '{key}' has invalid type {type(val)} for Pinecone",
            )
            if isinstance(val, list):
                self.assertTrue(all(isinstance(elem, str) for elem in val))

        # audio_rel_path must be "" instead of None
        self.assertEqual(metadata["audio_rel_path"], "")
        self.assertEqual(metadata["grid_used_str"], "2x3")
        self.assertIsInstance(metadata["tile_actions_json"], str)
        parsed_actions = json.loads(metadata["tile_actions_json"])
        self.assertEqual(len(parsed_actions), 2)
        self.assertEqual(parsed_actions[1]["grid_cells"], ["B2", "C3"])

    def test_build_extraction_prompt_respects_adaptive_grid(self) -> None:
        """Requirement 2: Verify that a 1x3 (3-tile) episode prompt instructs VLM

        to strictly populate tiles 1..3 and avoid empty slots.
        """
        record = IndexedEpisodeRecord(
            video_id="vid_adaptive",
            video_filename="driveway.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=5.0,
            end_time_sec=8.0,
            duration_sec=3.0,
            keyframe_times_sec=[5.0, 6.5, 8.0],
            roi_bbox={"x": 0, "y": 0, "width": 640, "height": 360},
            storyboard_rel_path="storyboards/event_001.jpg",
            audio_rel_path="audio/event_001.mp3",
            grid_used=(1, 3),  # 1x3 adaptive layout
        )

        prompt = build_extraction_prompt(record)

        # Must mention 1x3 active grid
        self.assertIn("1x3", prompt)
        self.assertIn("3 populated keyframe tiles", prompt)

        # Must explicitly constrain tile_actions to 1..3
        self.assertIn("1..3", prompt)
        self.assertIn("Never invent actions for empty/unpopulated tile slots", prompt)

        # Must include timestamps for tiles 1, 2, 3
        self.assertIn("Tile #1: +0.00s (abs 5.00s)", prompt)
        self.assertIn("Tile #2: +1.50s (abs 6.50s)", prompt)
        self.assertIn("Tile #3: +3.00s (abs 8.00s)", prompt)
        self.assertNotIn("Tile #4:", prompt)

    def test_extractor_multimodal_with_storyboard_and_audio(self) -> None:
        """Requirement 3: Mock Gemini client returning EpisodeExtractionSchema with shovel in B2, C3

        and verify apply_extraction updates fields and token counts.
        """
        # Create dummy storyboard image and audio file
        storyboards_dir = self.temp_dir / "storyboards"
        audio_dir = self.temp_dir / "audio"
        storyboards_dir.mkdir(parents=True, exist_ok=True)
        audio_dir.mkdir(parents=True, exist_ok=True)

        sb_path = storyboards_dir / "event_001.jpg"
        sb_path.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xff\xdb")

        aud_path = audio_dir / "event_001.mp3"
        aud_path.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00#TSSE\x00\x00\x00\x0f\x00\x00\x03Lavf58.29.100")

        record = IndexedEpisodeRecord(
            video_id="vid_shovel_event",
            video_filename="porch_cctv.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=10.0,
            end_time_sec=14.0,
            duration_sec=4.0,
            keyframe_times_sec=[10.0, 12.3, 14.0],
            roi_bbox={"x": 50, "y": 50, "width": 400, "height": 300},
            storyboard_rel_path="storyboards/event_001.jpg",
            audio_rel_path="audio/event_001.mp3",
            sieve_input_tokens=1120,
            grid_used=(1, 3),
        )

        fake_extraction = EpisodeExtractionSchema(
            episode_title="Snow Clearing with Metal Shovel",
            narrative_summary="A resident uses a metal scoop shovel to clear snow from the walkway.",
            actors=["resident"],
            objects_detected=["metal scoop shovel", "winter coat"],
            audio_transcript_and_cues="Rhythmic metallic scraping against stone steps.",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=0.0,
                    absolute_time_sec=10.0,
                    grid_cells=["B1"],
                    action_description="Resident steps onto walkway holding tool.",
                ),
                TileAction(
                    tile_number=2,
                    relative_time_sec=2.3,
                    absolute_time_sec=12.3,
                    grid_cells=["B2", "C3"],
                    action_description="Resident scoops snow with metal scoop shovel.",
                ),
                TileAction(
                    tile_number=3,
                    relative_time_sec=4.0,
                    absolute_time_sec=14.0,
                    grid_cells=["C3", "D4"],
                    action_description="Resident tosses snow to side lawn.",
                ),
            ],
        )

        # Mock Gemini client
        mock_response = MagicMock()
        mock_response.parsed = fake_extraction
        mock_response.text = fake_extraction.model_dump_json()
        mock_response.usage_metadata.prompt_token_count = 1122
        mock_response.usage_metadata.candidates_token_count = 168

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        extractor = StoryboardVLMExtractor(client=mock_client)
        updated = extractor.extract_episode(record=record, workspace_dir=self.temp_dir)

        # Verify client was called with correct model
        mock_client.models.generate_content.assert_called_once()
        call_kwargs = mock_client.models.generate_content.call_args[1]
        self.assertEqual(call_kwargs["model"], extractor.model_name)
        self.assertEqual(len(call_kwargs["contents"]), 3)  # Image Part, Audio Part, Text Part

        # Verify record was enriched
        self.assertEqual(updated.episode_title, "Snow Clearing with Metal Shovel")
        self.assertEqual(updated.sieve_input_tokens, 1122)
        self.assertEqual(updated.sieve_output_tokens, 168)
        self.assertIn("metal scoop shovel", updated.objects_detected)
        self.assertEqual(len(updated.tile_actions), 3)
        self.assertEqual(updated.tile_actions[1].grid_cells, ["B2", "C3"])
        self.assertEqual(updated.tile_actions[1].action_description, "Resident scoops snow with metal scoop shovel.")

    def test_extractor_updates_ingestion_receipt_on_disk(self) -> None:
        """Requirement 4: Verify extract_receipt updates ingestion_receipt.json

        on disk with extracted narratives and recalculated token counts.
        """
        sb_dir = self.temp_dir / "storyboards"
        sb_dir.mkdir(parents=True, exist_ok=True)
        (sb_dir / "event_001.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")
        (sb_dir / "event_002.jpg").write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF\x00")

        ep1 = IndexedEpisodeRecord(
            video_id="vid_disk_update",
            video_filename="camera.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=0.0,
            end_time_sec=3.0,
            duration_sec=3.0,
            storyboard_rel_path="storyboards/event_001.jpg",
            sieve_input_tokens=1120,
        )
        ep2 = IndexedEpisodeRecord(
            video_id="vid_disk_update",
            video_filename="camera.mp4",
            event_id=2,
            episode_index=2,
            start_time_sec=5.0,
            end_time_sec=8.0,
            duration_sec=3.0,
            storyboard_rel_path="storyboards/event_002.jpg",
            sieve_input_tokens=1120,
        )

        receipt = VideoIngestionReceipt(
            video_id="vid_disk_update",
            video_filename="camera.mp4",
            duration_sec=10.0,
            raw_frames_inspected=100,
            sampled_frames_decoded=100,
            storyboards_emitted=2,
            disturbances_suppressed=0,
            processing_wall_sec=0.45,
            estimated_direct_video_tokens=915,  # 10s * 91.5
            actual_sieve_rag_tokens=2240,
            tokens_saved_pct=0.0,
            episodes=[ep1, ep2],
            disturbances=[],
        )

        receipt_file = self.temp_dir / "ingestion_receipt.json"
        receipt_file.write_text(json.dumps(receipt.model_dump(), indent=2), encoding="utf-8")

        mock_extraction = EpisodeExtractionSchema(
            episode_title="Extracted Activity",
            narrative_summary="Activity summarized by VLM.",
            actors=["person"],
            objects_detected=[],
            audio_transcript_and_cues="",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[],
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_extraction
        mock_response.text = mock_extraction.model_dump_json()
        mock_response.usage_metadata.prompt_token_count = 1125
        mock_response.usage_metadata.candidates_token_count = 120

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        extractor = StoryboardVLMExtractor(client=mock_client)
        updated_receipt = extractor.extract_receipt(
            receipt=receipt,
            workspace_dir=self.temp_dir,
            save_updated_receipt=True,
        )

        # Check in-memory receipt updates
        self.assertEqual(updated_receipt.actual_sieve_rag_tokens, 2250)  # 1125 + 1125

        # Check persisted file on disk
        self.assertTrue(receipt_file.is_file())
        disk_data = json.loads(receipt_file.read_text(encoding="utf-8"))
        self.assertEqual(disk_data["actual_sieve_rag_tokens"], 2250)
        self.assertEqual(disk_data["episodes"][0]["episode_title"], "Extracted Activity")
        self.assertEqual(disk_data["episodes"][1]["episode_title"], "Extracted Activity")

    def test_vector_index_upsert_and_semantic_search_with_filters(self) -> None:
        """Requirement 5: Index 3 distinct episodes, query for shovel near doorway,

        verify shovel episode ranks #1 and metadata filters operate accurately.
        """
        local_store_path = self.temp_dir / "vector_index.json"
        embedder = EpisodeEmbedder(use_mock_embeddings=True)
        vindex = SieveRAGVectorIndex(
            embedder=embedder,
            local_index_path=local_store_path,
        )

        # Episode 1: Normal package delivery
        ep1 = IndexedEpisodeRecord(
            video_id="vid_alpha",
            video_filename="porch_day.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=0.0,
            end_time_sec=4.0,
            duration_sec=4.0,
            storyboard_rel_path="storyboards/event_001.jpg",
            episode_title="Routine Package Delivery",
            narrative_summary="Delivery courier in uniform drops off a brown cardboard parcel at the front door.",
            actors=["courier"],
            objects_detected=["brown cardboard parcel", "handheld scanner"],
            audio_transcript_and_cues="Quiet footsteps and door mat rustle.",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=0.0,
                    absolute_time_sec=0.0,
                    grid_cells=["A1"],
                    action_description="Courier walks toward door.",
                )
            ],
        )

        # Episode 2: Resident with a snow shovel
        ep2 = IndexedEpisodeRecord(
            video_id="vid_alpha",
            video_filename="porch_day.mp4",
            event_id=2,
            episode_index=2,
            start_time_sec=10.0,
            end_time_sec=16.0,
            duration_sec=6.0,
            storyboard_rel_path="storyboards/event_002.jpg",
            episode_title="Walkway Snow Shoveling",
            narrative_summary="Resident holding a long metal shovel clearing snow near the doorway entrance.",
            actors=["resident"],
            objects_detected=["long metal shovel", "boots"],
            audio_transcript_and_cues="Scraping metal sound against concrete.",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=1.5,
                    absolute_time_sec=11.5,
                    grid_cells=["B2", "C3"],
                    action_description="Resident pushes long metal shovel along doorway entrance.",
                )
            ],
        )

        # Episode 3: Stray cat on porch (flagged as anomaly with high severity for filter testing)
        ep3 = IndexedEpisodeRecord(
            video_id="vid_beta",
            video_filename="night_feed.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=20.0,
            end_time_sec=25.0,
            duration_sec=5.0,
            storyboard_rel_path="storyboards/event_003.jpg",
            episode_title="Nocturnal Animal Intrusion",
            narrative_summary="A black stray cat jumps onto the porch railing at night knocking over a flower pot.",
            actors=["stray cat"],
            objects_detected=["porch railing", "flower pot"],
            audio_transcript_and_cues="Ceramic crash and sudden feline hiss.",
            anomaly_or_security_flag=True,
            severity="high",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=0.5,
                    absolute_time_sec=20.5,
                    grid_cells=["D4"],
                    action_description="Cat leaps onto porch railing.",
                )
            ],
        )

        # Upsert records
        upsert_res = vindex.upsert_episodes([ep1, ep2, ep3])
        self.assertEqual(upsert_res["upserted_count"], 3)
        self.assertTrue(local_store_path.is_file())

        # Query for the shovel activity
        query_text = "person holding a long metal shovel near the doorway"
        hits = vindex.search(query=query_text, top_k=3)

        self.assertEqual(len(hits), 3)
        # Verify Shovel Episode is ranked #1 with highest score
        self.assertEqual(hits[0]["episode_title"], "Walkway Snow Shoveling")
        self.assertIn("long metal shovel", hits[0]["objects_detected"])
        self.assertGreater(hits[0]["score"], hits[1]["score"])
        self.assertGreater(hits[0]["score"], hits[2]["score"])

        # Test video_id filter
        hits_beta = vindex.search(query=query_text, top_k=3, video_id="vid_beta")
        self.assertEqual(len(hits_beta), 1)
        self.assertEqual(hits_beta[0]["video_id"], "vid_beta")
        self.assertEqual(hits_beta[0]["episode_title"], "Nocturnal Animal Intrusion")

        # Test anomaly_only filter
        hits_anomaly = vindex.search(query=query_text, top_k=3, anomaly_only=True)
        self.assertEqual(len(hits_anomaly), 1)
        self.assertTrue(hits_anomaly[0]["anomaly_or_security_flag"])
        self.assertEqual(hits_anomaly[0]["episode_title"], "Nocturnal Animal Intrusion")

        # Test min_severity filter ("high")
        hits_high_sev = vindex.search(query=query_text, top_k=3, min_severity="high")
        self.assertEqual(len(hits_high_sev), 1)
        self.assertEqual(hits_high_sev[0]["severity"], "high")

    def test_end_to_end_phase1_to_phase2_pipeline(self) -> None:
        """Requirement 6: Complete end-to-end integration:

        1. Generate synthetic video
        2. Run Phase 1 SieveVideoIngestor.ingest_video()
        3. Run Phase 2 StoryboardVLMExtractor.extract_receipt() with mock Gemini
        4. Upsert into SieveRAGVectorIndex
        5. Query and retrieve the episode.
        """
        # 1. Generate synthetic video (3 seconds at 10 fps with moving object)
        vid_path = self.temp_dir / "pipeline_test.mp4"
        h, w = 360, 640
        out = cv2.VideoWriter(
            str(vid_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            10.0,
            (w, h),
        )
        for i in range(30):
            frame = np.zeros((h, w, 3), dtype=np.uint8)
            if 10 <= i <= 24:
                # Moving bright square
                cv2.rectangle(frame, (100 + i * 15, 150), (160 + i * 15, 210), (0, 255, 255), -1)
            out.write(frame)
        out.release()

        # 2. Run Phase 1 Ingestion
        ingest_output_dir = self.temp_dir / "ingested_run"
        ingestor = SieveVideoIngestor(
            sample_fps=10.0,
            pre_roll_sec=1.0,
            merge_gap_sec=2.0,
            tile_max_dim=640,
        )
        receipt_dict, events, disturbances, savings = ingestor.ingest_video(
            video_path=vid_path,
            output_dir=ingest_output_dir,
            video_id="e2e_vid_001",
        )
        receipt = VideoIngestionReceipt.model_validate(receipt_dict)
        self.assertGreater(len(receipt.episodes), 0)

        # 3. Run Phase 2 Extraction with mock Gemini
        mock_extraction = EpisodeExtractionSchema(
            episode_title="Technician Relocating Yellow Inspection Box",
            narrative_summary="A technician in safety gear carries a yellow inspection box across the laboratory floor.",
            actors=["technician"],
            objects_detected=["yellow inspection box", "safety goggles"],
            audio_transcript_and_cues="Quiet machinery hum in background.",
            anomaly_or_security_flag=False,
            severity="low",
            tile_actions=[
                TileAction(
                    tile_number=1,
                    relative_time_sec=0.0,
                    absolute_time_sec=1.0,
                    grid_cells=["B2"],
                    action_description="Technician lifts yellow box.",
                ),
                TileAction(
                    tile_number=2,
                    relative_time_sec=1.2,
                    absolute_time_sec=2.2,
                    grid_cells=["C3"],
                    action_description="Technician walks box across lab.",
                ),
            ],
        )
        mock_response = MagicMock()
        mock_response.parsed = mock_extraction
        mock_response.text = mock_extraction.model_dump_json()
        mock_response.usage_metadata.prompt_token_count = 1120
        mock_response.usage_metadata.candidates_token_count = 130

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        extractor = StoryboardVLMExtractor(client=mock_client)
        enriched_receipt = extractor.extract_receipt(
            receipt=receipt,
            workspace_dir=ingest_output_dir,
            save_updated_receipt=True,
        )

        self.assertEqual(enriched_receipt.episodes[0].episode_title, "Technician Relocating Yellow Inspection Box")

        # 4. Index enriched episodes
        local_vec_path = ingest_output_dir / "vector_index.json"
        embedder = EpisodeEmbedder(use_mock_embeddings=True)
        vindex = SieveRAGVectorIndex(
            embedder=embedder,
            local_index_path=local_vec_path,
        )
        upsert_res = vindex.upsert_episodes(enriched_receipt.episodes)
        self.assertEqual(upsert_res["upserted_count"], len(enriched_receipt.episodes))

        # 5. Query vector index
        query_hits = vindex.search(
            query="technician carrying a yellow inspection box",
            top_k=2,
        )

        self.assertGreater(len(query_hits), 0)
        best_hit = query_hits[0]
        self.assertEqual(best_hit["episode_title"], "Technician Relocating Yellow Inspection Box")
        self.assertIn("yellow inspection box", best_hit["objects_detected"])
        self.assertEqual(best_hit["video_id"], "e2e_vid_001")
        self.assertEqual(len(best_hit["tile_actions"]), 2)
        self.assertEqual(best_hit["tile_actions"][0]["grid_cells"], ["B2"])

    def test_vector_index_pinecone_mode_upsert_and_query(self) -> None:
        """Verify Pinecone mode initialization, batch upsert, and query filtering using mock Pinecone client."""
        mock_pc = MagicMock()
        mock_index = MagicMock()

        # Mock list_indexes
        idx_info = MagicMock()
        idx_info.name = "sieverag-test-index"
        mock_pc.list_indexes.return_value = [idx_info]
        mock_pc.Index.return_value = mock_index

        # Mock query return
        match_item = MagicMock()
        match_item.id = "vid_pc_001_ep_001"
        match_item.score = 0.92
        match_item.metadata = {
            "video_id": "vid_pc_001",
            "episode_index": 1,
            "start_time_sec": 1.0,
            "end_time_sec": 4.0,
            "episode_title": "Person Approaching Gate",
            "narrative_summary": "Individual walks towards perimeter gate.",
            "actors": ["individual"],
            "objects_detected": ["gate"],
            "audio_transcript_and_cues": "",
            "anomaly_or_security_flag": True,
            "severity": "medium",
            "storyboard_rel_path": "storyboards/event_001.jpg",
            "audio_rel_path": "",
            "grid_used_str": "2x3",
            "tile_actions_json": json.dumps([{"tile_number": 1, "action_description": "Walking"}]),
        }
        query_response = MagicMock()
        query_response.matches = [match_item]
        mock_index.query.return_value = query_response

        embedder = EpisodeEmbedder(use_mock_embeddings=True)
        vindex = SieveRAGVectorIndex(
            embedder=embedder,
            pinecone_client=mock_pc,
            pinecone_index_name="sieverag-test-index",
            local_index_path=self.temp_dir / "pc_local_mirror.json",
        )

        self.assertTrue(vindex.has_pinecone)

        # Upsert
        rec = IndexedEpisodeRecord(
            video_id="vid_pc_001",
            video_filename="gate.mp4",
            event_id=1,
            episode_index=1,
            start_time_sec=1.0,
            end_time_sec=4.0,
            duration_sec=3.0,
            storyboard_rel_path="storyboards/event_001.jpg",
            episode_title="Person Approaching Gate",
            anomaly_or_security_flag=True,
            severity="medium",
        )
        res = vindex.upsert_episodes([rec], namespace="security_dept")
        self.assertEqual(res["backend"], "pinecone")
        mock_index.upsert.assert_called_once()
        upsert_kwargs = mock_index.upsert.call_args[1]
        self.assertEqual(upsert_kwargs["namespace"], "security_dept")
        self.assertEqual(len(upsert_kwargs["vectors"]), 1)

        # Search with filters
        hits = vindex.search(
            query="individual approaching perimeter gate",
            top_k=1,
            video_id="vid_pc_001",
            anomaly_only=True,
            min_severity="medium",
            namespace="security_dept",
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["episode_title"], "Person Approaching Gate")
        self.assertEqual(hits[0]["score"], 0.92)

        # Check filter passed to Pinecone
        mock_index.query.assert_called_once()
        query_kwargs = mock_index.query.call_args[1]
        self.assertEqual(query_kwargs["namespace"], "security_dept")
        self.assertIn("video_id", query_kwargs["filter"])
        self.assertIn("anomaly_or_security_flag", query_kwargs["filter"])
        self.assertIn("severity", query_kwargs["filter"])


if __name__ == "__main__":
    unittest.main()
