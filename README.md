# SieveRAG: Temporal Video RAG & Multimodal Intelligence System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![vlm-sieve 0.3.0](https://img.shields.io/badge/vlm--sieve-v0.3.0-indigo.svg)](https://github.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Google GenAI](https://img.shields.io/badge/Google_GenAI-Gemini_Flash-4285F4.svg)](https://ai.google.dev/)
[![Pinecone v5](https://img.shields.io/badge/Pinecone-Serverless-000000.svg)](https://www.pinecone.io/)
[![Tests](https://img.shields.io/badge/tests-26%20passed-brightgreen.svg)]()

**SieveRAG** is a high-performance, token-efficient **Temporal Video Retrieval-Augmented Generation (RAG)** platform. Built on the open-source [`vlm-sieve`](https://github.com/) (`v0.3.0`) engine, SieveRAG eliminates the immense token overhead of raw video streams by transforming long-form video into motion-gated, disturbance-suppressed storyboards with synchronized audio slices, indexing them into vector storage, and providing grounded question-answering with exact temporal and spatial citations.

---

## The Problem: The Raw Video Token Dilemma

Feeding direct video frames into Vision-Language Models (VLMs) is prohibitively expensive and slow:
- **Direct Video Ingestion:** Consumes approximately **91.5 tokens/second** (5,490 tokens/minute, or ~330,000 tokens/hour).
- **Disturbances & Clutter:** Shaking tree leaves, lighting shifts, digital camera noise, and static backgrounds waste up to **80% of VLM compute**.
- **Temporal Blindness:** Standard VLMs struggle to localize the exact sub-second timestamp or spatial region where an action occurred.

### The SieveRAG Solution
SieveRAG combines deterministic computer vision motion-gating with adaptive storyboard generation and modern multimodal RAG:
1. **85% – 95% Token Reduction:** Filters out static time and minor environmental disturbances (foliage, headlights) before invoking any LLM or VLM.
2. **Adaptive Grid Gating:** Automatically selects optimal tile geometry (`1x1`, `1x2`, `2x2`, `2x3`, etc.) with spatial grid overlays (`A1..D4`) and burned millisecond timestamps.
3. **Synchronized Audio Slicing:** Extracts lightweight mono MP3 slices (`16kHz`, `32kbps`) corresponding exactly to detected visual activity periods.
4. **Grounded Multimodal Synthesis:** Queries Gemini with relevant storyboard frames and audio cues, providing answers with clickable temporal and spatial citations that seek an HTML5 video stream in real-time.

---

## System Architecture

```mermaid
flowchart TD
    subgraph Ingestion["Phase 1: Ingestion & Gating"]
        V[Raw Video MP4/MOV] --> Sieve[vlm-sieve v0.3.0 Engine]
        Sieve --> Motion[Adaptive Motion Gating]
        Sieve --> Dist[Disturbance Suppression]
        Motion --> Storyboards[Storyboard Canvases JPEG]
        Motion --> AudioSlice[FFmpeg 16kHz Sliced MP3]
    end

    subgraph Extraction["Phase 2: Multimodal Extraction & Indexing"]
        Storyboards --> GeminiVLM[Gemini Multimodal VLM]
        AudioSlice --> GeminiVLM
        GeminiVLM --> Schema[EpisodeExtractionSchema]
        Schema --> Embedder[text-embedding-004]
        Embedder --> VectorDB[(Pinecone / Local JSON Mirror)]
    end

    subgraph Synthesis["Phase 3 & 4: RAG Engine & Command Center"]
        UserQuery[User Question / Query] --> Embedder
        VectorDB --> TopK[Top-K Semantic Episodes]
        TopK --> ReVerify[Visual Re-verification: Top Storyboards]
        ReVerify --> RAGSynth[SieveRAGSynthesizer]
        RAGSynth --> Citations[Temporal & Spatial Citations]
        Citations --> WebUI[Command Center UI]
        WebUI --> VideoStream[HTML5 Click-to-Seek Video Streaming]
    end
```

---

## Key Features

- **vlm-sieve v0.3.0 Integration:** Full support for `target_vlm` engines:
  - `gemini`: Dynamic storyboard canvas packing with spatial coordinate overlays.
  - `gpt-4o`: Tile sizes aligned to 512x512 grids.
  - `qwen-vl`: Enforces strict **Patch-28 alignment** (`width % 28 == 0` and `height % 28 == 0`).
  - `local-7b`: Optimized for edge deployment and GGUF / Ollama models with telemetry tracking KV cache savings in MB.
- **Official Google GenAI SDK:** Uses the modern `from google import genai` SDK (`google-genai`) with structured JSON schema outputs (`EpisodeExtractionSchema`, `RAGAnswerSchema`).
- **Pinecone Serverless + Local Mirror:** Connects to Pinecone Serverless indices (`cosine` metric, 768 dimensions), with transparent, zero-dependency offline NumPy cosine similarity fallback when no API key is provided.
- **Two-Stage Visual Re-Verification:** During question-answering, SieveRAG fetches the top storyboard JPEGs from disk and sends them directly to Gemini alongside the semantic transcripts to eliminate hallucinations.
- **Interactive Command Center UI:** A dark-themed dashboard (`#0b0f17`) featuring:
  - Live aggregate corpus telemetry pills (total duration, storyboards emitted, disturbances suppressed, token savings %).
  - Video library management and drag-and-drop file/path ingestion.
  - HTML5 video player with sub-second click-to-seek timeline scrubber.
  - Active episode inspector displaying storyboard frames, audio slice player, and itemized tile actions.
  - Temporal RAG console with clickable citation chips (`[Ep #1 • Tile #2 • 14.20s • Grid B2]`) that immediately jump the video player to the exact moment.

---

## Repository Structure

```
TemporalVideo-RAG/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py         # Configuration singleton & environment resolution
│   │   ├── schemas.py        # Pydantic v2 data models (TileAction, Receipts, Citations)
│   │   ├── ingestor.py       # SieveVideoIngestor (vlm-sieve + FFmpeg audio slicing)
│   │   ├── extractor.py      # StoryboardVLMExtractor (Gemini multimodal extraction)
│   │   ├── indexer.py        # EpisodeEmbedder & SieveRAGVectorIndex (Pinecone/Local)
│   │   ├── rag.py            # SieveRAGSynthesizer (Prompt builder & visual re-verification)
│   │   └── main.py           # FastAPI REST application & media static router
│   ├── data/
│   │   └── storage/          # Storage root for videos, storyboards, audio, and receipts
│   └── requirements.txt      # Backend Python dependencies
├── frontend/
│   ├── index.html            # Command Center Single Page Application
│   ├── styles.css            # Obsidian/Slate dark UI design system
│   └── app.js                # State management, timeline scrubber, and RAG interaction
├── tests/
│   ├── test_ingestion.py                     # Phase 1 Ingestion test suite (11 tests)
│   ├── test_phase2_extraction_and_indexing.py # Phase 2 Extraction & Indexing (7 tests)
│   ├── test_phase3_rag_and_api.py            # Phase 3 RAG & API test suite (4 tests)
│   └── test_phase4_ui_and_e2e.py             # Phase 4 UI, Streaming & Telemetry (4 tests)
└── README.md
```

---

## Token Economy & Telemetry Benchmarks

| Metric | Direct Raw Video | SieveRAG (Gemini) | SieveRAG (Qwen-VL) | Savings % |
| :--- | :--- | :--- | :--- | :--- |
| **Tokens per 60s Video** | 5,490 tokens | 480 – 960 tokens | 640 – 1,120 tokens | **82% – 91%** |
| **Tokens per 1hr Stream** | 329,400 tokens | 28,000 – 52,000 tokens | 38,000 – 67,000 tokens | **84% – 92%** |
| **Qwen-7B KV Cache VRAM** | ~14,200 MB | ~1,250 MB | ~1,580 MB | **89% VRAM Saved** |
| **Spatial Precision** | Coarse / None | Exact `A1..D4` Grid Cells | Exact `A1..D4` Grid Cells | **Sub-frame Grounding** |

---

## REST API Reference

### 1. Ingestion & Video Management
- `POST /api/videos/ingest`: Ingest video via multipart file upload or local filesystem path.
  - **Form Fields:** `file` (optional), `local_video_path` (optional), `target_vlm` (`gemini` | `gpt-4o` | `qwen-vl` | `local-7b`), `run_vlm_extraction` (bool), `index_episodes` (bool).
  - **Returns:** Complete `VideoIngestionReceipt` with `video_url`, `episodes`, token metrics, and disturbances.
- `GET /api/videos`: List all ingested video summary records.
- `GET /api/videos/{video_id}`: Retrieve detailed receipt with resolved `/media/...` URLs.
- `DELETE /api/videos/{video_id}`: Purge video from disk and remove all indexed vector embeddings.

### 2. Retrieval & Grounded RAG
- `POST /api/search`: Perform semantic temporal vector search across indexed episodes.
  - **Payload:** `{"query": "forklift moving pallets", "top_k": 4, "video_id": null, "anomaly_only": false}`
  - **Returns:** Ranked list of episode matches with cosine similarity scores and timing.
- `POST /api/ask`: Synthesize grounded answer with citations and optional visual re-verification.
  - **Payload:** `{"question": "What security incident occurred?", "top_k": 4, "include_storyboards": true}`
  - **Returns:** Grounded answer narrative, confidence rating, security alert trigger, and itemized citations with seek timestamps.

### 3. System & Telemetry
- `GET /api/health`: Service health, configured `vlmsieve` version, backend type (`pinecone` or `local`), and indexed episode count.
- `GET /api/stats`: Aggregate corpus telemetry (total duration, frames inspected, storyboards emitted, disturbances suppressed, token savings %).

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.12)
- FFmpeg installed or available via `imageio-ffmpeg`

### 2. Installation
```bash
# Clone the repository
git clone https://github.com/your-username/TemporalVideo-RAG.git
cd TemporalVideo-RAG

# Create and activate virtual environment
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Configuration (.env)
Create a `.env` file in the project root:
```env
# Google Gemini API Key (optional for offline testing; required for live VLM extraction)
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_VLM_MODEL=gemini-2.0-flash
GEMINI_EMBEDDING_MODEL=text-embedding-004

# Pinecone Serverless (optional; defaults to local JSON vector mirror if empty)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=sieverag-episodes
PINECONE_NAMESPACE=default

# Defaults
DEFAULT_TARGET_VLM=gemini
STORAGE_DIR=backend/data/storage
```

### 4. Running the Application
Start the FastAPI server:
```bash
uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open your browser at **`http://localhost:8000`** to access the **SieveRAG Command Center UI**.

### 5. Running the Complete Test Suite
Run the 26 unit and end-to-end integration tests:
```bash
pytest -v
```

---

## License

This project is licensed under the Apache 2.0 License.

