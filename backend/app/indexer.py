"""Vector indexing and temporal semantic retrieval engine for SieveRAG.

Provides hybrid storage supporting Pinecone Serverless cloud indexing and a
deterministic local JSON + NumPy cosine vector store fallback for offline/local environments.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from google import genai
from google.genai import types

from backend.app.config import Settings, get_settings
from backend.app.schemas import IndexedEpisodeRecord

logger = logging.getLogger("sieverag.indexer")


class EpisodeEmbedder:
    """Multimodal text embedder producing dense semantic vectors via Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        dimension: int | None = None,
        client: Any | None = None,
        use_mock_embeddings: bool = False,
        settings: Settings | None = None,
    ) -> None:
        """Initialize EpisodeEmbedder.

        Args:
            api_key: Optional Gemini API key.
            model_name: Target embedding model (defaults to GEMINI_EMBEDDING_MODEL).
            dimension: Embedding vector dimension (defaults to EMBEDDING_DIMENSION = 768).
            client: Optional injected genai.Client instance for testing.
            use_mock_embeddings: If True, uses deterministic hashing without network calls.
            settings: Optional application settings.
        """
        self.settings = settings or get_settings()
        self.api_key = api_key or self.settings.GEMINI_API_KEY
        self.model_name = model_name or self.settings.GEMINI_EMBEDDING_MODEL
        self.dimension = dimension or self.settings.EMBEDDING_DIMENSION
        self.client = client
        self.use_mock_embeddings = use_mock_embeddings

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        key = self.api_key or self.settings.GEMINI_API_KEY
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not configured for text embedding. "
                "Set GEMINI_API_KEY or instantiate with use_mock_embeddings=True."
            )
        self.client = genai.Client(api_key=key)
        return self.client

    @staticmethod
    def deterministic_hash_embedding(text: str, dimension: int = 768) -> list[float]:
        """Compute a deterministic, L2-normalized dense vector using token and n-gram hashing.

        Ensures semantically related texts sharing keywords (e.g. 'shovel', 'snow', 'doorway')
        exhibit high cosine similarity without requiring external API calls.

        Args:
            text: Input text string to embed.
            dimension: Target vector dimensionality (default 768).

        Returns:
            Unit-normalized float list of length dimension.
        """
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            vec = [0.0] * dimension
            vec[0] = 1.0
            return vec

        vec = [0.0] * dimension

        # Combine unigrams and bigrams
        features = list(tokens)
        for i in range(len(tokens) - 1):
            features.append(f"{tokens[i]}_{tokens[i + 1]}")

        for feat in features:
            digest = hashlib.sha256(feat.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % dimension
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0.0:
            return [x / norm for x in vec]
        vec[0] = 1.0
        return vec

    def embed_text(
        self,
        text: str,
        task_type: str = "RETRIEVAL_DOCUMENT",
    ) -> list[float]:
        """Generate a dense embedding vector for text.

        Args:
            text: The text content to embed.
            task_type: Embedding task type ("RETRIEVAL_DOCUMENT" or "RETRIEVAL_QUERY").

        Returns:
            Dense float vector of dimensionality self.dimension.
        """
        if self.use_mock_embeddings:
            return self.deterministic_hash_embedding(text, dimension=self.dimension)

        # Attempt live Gemini embedding if key or client is available
        key = self.api_key or self.settings.GEMINI_API_KEY
        if self.client is not None or key:
            try:
                client = self._get_client()
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimension,
                    ),
                )
                if hasattr(response, "embeddings") and response.embeddings:
                    values = getattr(response.embeddings[0], "values", None)
                    if values:
                        return list(values)
            except Exception as exc:
                logger.warning(
                    "Gemini embed_content call failed (%s); falling back to deterministic hashing.",
                    exc,
                )

        # Fallback to deterministic hash embedding
        return self.deterministic_hash_embedding(text, dimension=self.dimension)


class SieveRAGVectorIndex:
    """Hybrid Vector Store combining Pinecone Serverless and Local JSON/NumPy storage."""

    def __init__(
        self,
        embedder: EpisodeEmbedder | None = None,
        pinecone_api_key: str | None = None,
        pinecone_index_name: str | None = None,
        pinecone_client: Any | None = None,
        local_index_path: str | Path | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize SieveRAGVectorIndex.

        Args:
            embedder: EpisodeEmbedder instance.
            pinecone_api_key: Pinecone API key.
            pinecone_index_name: Target Pinecone index name.
            pinecone_client: Injected Pinecone client instance (for testing/mocking).
            local_index_path: Path to local JSON vector store backup/mirror.
            settings: Optional application settings.
        """
        self.settings = settings or get_settings()
        self.embedder = embedder or EpisodeEmbedder(settings=self.settings)

        self.pinecone_api_key = pinecone_api_key or self.settings.PINECONE_API_KEY
        self.pinecone_index_name = pinecone_index_name or self.settings.PINECONE_INDEX_NAME
        self.pinecone_client = pinecone_client

        # Local storage mirror path
        if local_index_path:
            self.local_index_path = Path(local_index_path).resolve()
        else:
            self.local_index_path = (self.settings.STORAGE_DIR / "vector_index.json").resolve()
        self.local_index_path.parent.mkdir(parents=True, exist_ok=True)

        self.local_store: dict[str, dict[str, Any]] = self._load_local_store()

        # Initialize Pinecone connection if credentials or client provided
        self.pinecone_index: Any | None = None
        self._init_pinecone()

    @property
    def has_pinecone(self) -> bool:
        """Whether a live Pinecone index connection is active."""
        return self.pinecone_index is not None

    def _init_pinecone(self) -> None:
        """Connect to or create the Pinecone index if credentials exist."""
        if not self.pinecone_api_key and self.pinecone_client is None:
            logger.info("Pinecone API key not provided; operating in Local JSON/NumPy vector mode.")
            return

        try:
            from pinecone import Pinecone, ServerlessSpec

            pc = self.pinecone_client or Pinecone(api_key=self.pinecone_api_key)

            # Check existing indexes
            existing_indexes = [idx.name for idx in pc.list_indexes()]
            if self.pinecone_index_name not in existing_indexes:
                logger.info(
                    "Creating Serverless Pinecone index '%s' (dim=%d, metric=cosine)",
                    self.pinecone_index_name,
                    self.embedder.dimension,
                )
                pc.create_index(
                    name=self.pinecone_index_name,
                    dimension=self.embedder.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )

            self.pinecone_index = pc.Index(self.pinecone_index_name)
            logger.info("Connected to Pinecone index '%s'", self.pinecone_index_name)
        except Exception as exc:
            logger.warning("Failed to initialize Pinecone (%s); falling back to Local mode.", exc)
            self.pinecone_index = None

    def _load_local_store(self) -> dict[str, dict[str, Any]]:
        """Load the local JSON vector store from disk."""
        if self.local_index_path.is_file():
            try:
                data = json.loads(self.local_index_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                logger.warning("Could not read local vector index %s: %s", self.local_index_path, exc)
        return {}

    def _save_local_store(self) -> None:
        """Persist the local JSON vector store to disk."""
        try:
            self.local_index_path.write_text(
                json.dumps(self.local_store, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save local vector store to %s: %s", self.local_index_path, exc)

    def upsert_episodes(
        self,
        records: list[IndexedEpisodeRecord],
        namespace: str | None = None,
    ) -> dict[str, Any]:
        """Index episode records into Pinecone and the local JSON vector mirror.

        Args:
            records: List of IndexedEpisodeRecord instances to embed and store.
            namespace: Target namespace (defaults to settings.PINECONE_NAMESPACE).

        Returns:
            Dictionary containing upserted_count, vector_ids, and backend used.
        """
        ns = namespace or self.settings.PINECONE_NAMESPACE
        vector_ids: list[str] = []
        pinecone_vectors: list[dict[str, Any]] = []

        for record in records:
            vec_id = f"{record.video_id}_ep_{record.episode_index:03d}"
            vector_ids.append(vec_id)

            # Compute dense embedding
            embed_text = record.to_embedding_text()
            vec_values = self.embedder.embed_text(embed_text, task_type="RETRIEVAL_DOCUMENT")

            # Extract flat Pinecone metadata
            metadata = record.to_vector_metadata()

            # Store in local mirror
            self.local_store[vec_id] = {
                "id": vec_id,
                "values": vec_values,
                "metadata": metadata,
                "namespace": ns,
            }

            pinecone_vectors.append({
                "id": vec_id,
                "values": vec_values,
                "metadata": metadata,
            })

        # Save local JSON mirror
        self._save_local_store()

        # Upsert to Pinecone if connected
        backend_used = "local"
        if self.has_pinecone:
            try:
                # Upsert in batches of 100
                batch_size = 100
                for i in range(0, len(pinecone_vectors), batch_size):
                    batch = pinecone_vectors[i : i + batch_size]
                    self.pinecone_index.upsert(vectors=batch, namespace=ns)
                backend_used = "pinecone"
                logger.info("Upserted %d vectors to Pinecone namespace '%s'", len(records), ns)
            except Exception as exc:
                logger.warning("Pinecone upsert failed (%s); saved to local index mirror.", exc)

        return {
            "upserted_count": len(records),
            "vector_ids": vector_ids,
            "backend": backend_used,
        }

    def delete_video(self, video_id: str, namespace: str | None = None) -> int:
        """Remove all indexed vector embeddings matching video_id.

        Deletes from local JSON store and from Pinecone if connected.

        Args:
            video_id: The identifier of the video whose episode vectors are to be deleted.
            namespace: Optional namespace (defaults to settings.PINECONE_NAMESPACE).

        Returns:
            Number of removed vector records.
        """
        ns = namespace or self.settings.PINECONE_NAMESPACE
        keys_to_delete = [
            k for k, item in self.local_store.items()
            if item.get("metadata", {}).get("video_id") == video_id
            or k.startswith(f"{video_id}_ep_")
        ]

        removed_count = len(keys_to_delete)
        for k in keys_to_delete:
            del self.local_store[k]

        if removed_count > 0:
            self._save_local_store()

        if self.has_pinecone:
            try:
                if keys_to_delete:
                    self.pinecone_index.delete(ids=keys_to_delete, namespace=ns)
                else:
                    self.pinecone_index.delete(filter={"video_id": {"$eq": video_id}}, namespace=ns)
            except Exception as exc:
                logger.warning("Pinecone vector deletion failed for %s: %s", video_id, exc)

        logger.info("Deleted %d vectors for video_id '%s'", removed_count, video_id)
        return removed_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        video_id: str | None = None,
        min_severity: str | None = None,
        anomaly_only: bool = False,
        namespace: str | None = None,
    ) -> list[dict[str, Any]]:
        """Perform semantic vector retrieval over indexed video episodes.

        Supports metadata filtering by video_id, anomaly_only, and minimum severity.

        Args:
            query: Natural language query text.
            top_k: Maximum number of ranked hits to return.
            video_id: Optional filter for a specific video ID.
            min_severity: Optional filter: 'low', 'medium', or 'high'.
            anomaly_only: If True, only returns records with anomaly_or_security_flag = True.
            namespace: Target namespace (defaults to settings.PINECONE_NAMESPACE).

        Returns:
            Ranked list of hit dictionaries containing metadata and tile_actions.
        """
        ns = namespace or self.settings.PINECONE_NAMESPACE
        query_vec = self.embedder.embed_text(query, task_type="RETRIEVAL_QUERY")

        # Try querying Pinecone first if available
        if self.has_pinecone:
            try:
                filt: dict[str, Any] = {}
                if video_id:
                    filt["video_id"] = {"$eq": video_id}
                if anomaly_only:
                    filt["anomaly_or_security_flag"] = {"$eq": True}
                if min_severity == "high":
                    filt["severity"] = {"$eq": "high"}
                elif min_severity == "medium":
                    filt["severity"] = {"$in": ["medium", "high"]}

                res = self.pinecone_index.query(
                    vector=query_vec,
                    top_k=top_k,
                    namespace=ns,
                    include_metadata=True,
                    filter=filt if filt else None,
                )

                matches = getattr(res, "matches", []) or res.get("matches", [])
                hits: list[dict[str, Any]] = []
                for m in matches:
                    vec_id = getattr(m, "id", None) or m.get("id")
                    score = getattr(m, "score", None) or m.get("score", 0.0)
                    meta = getattr(m, "metadata", None) or m.get("metadata", {})
                    hits.append(self._format_hit(vec_id=vec_id, score=score, meta=meta))
                return hits
            except Exception as exc:
                logger.warning("Pinecone query failed (%s); falling back to Local index.", exc)

        # Local NumPy Cosine Similarity Search
        return self._search_local(
            query_vec=query_vec,
            top_k=top_k,
            video_id=video_id,
            min_severity=min_severity,
            anomaly_only=anomaly_only,
            namespace=ns,
        )

    def _search_local(
        self,
        query_vec: list[float],
        top_k: int,
        video_id: str | None,
        min_severity: str | None,
        anomaly_only: bool,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """Execute local cosine similarity search over local_store."""
        q = np.array(query_vec, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return []

        scored_candidates: list[tuple[float, str, dict[str, Any]]] = []

        for vec_id, item in self.local_store.items():
            if item.get("namespace") != namespace:
                continue

            meta = item.get("metadata", {})

            # Filter checks
            if video_id and meta.get("video_id") != video_id:
                continue
            if anomaly_only and not meta.get("anomaly_or_security_flag", False):
                continue
            if min_severity == "high" and meta.get("severity") != "high":
                continue
            if min_severity == "medium" and meta.get("severity") not in ("medium", "high"):
                continue

            v = np.array(item.get("values", []), dtype=np.float32)
            v_norm = np.linalg.norm(v)
            if v_norm == 0.0:
                continue

            score = float(np.dot(q, v) / (q_norm * v_norm))
            scored_candidates.append((score, vec_id, meta))

        # Sort descending by cosine similarity score
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        hits: list[dict[str, Any]] = []
        for score, vec_id, meta in scored_candidates[:top_k]:
            hits.append(self._format_hit(vec_id=vec_id, score=score, meta=meta))

        return hits

    def _format_hit(self, vec_id: str, score: float, meta: dict[str, Any]) -> dict[str, Any]:
        """Format raw vector metadata and score into a clean hit record."""
        tile_actions_raw = meta.get("tile_actions_json", "[]")
        tile_actions: list[dict[str, Any]] = []
        if isinstance(tile_actions_raw, str):
            try:
                tile_actions = json.loads(tile_actions_raw)
            except Exception:
                tile_actions = []
        elif isinstance(tile_actions_raw, list):
            tile_actions = tile_actions_raw

        return {
            "vector_id": str(vec_id),
            "score": round(float(score), 4),
            "video_id": str(meta.get("video_id", "")),
            "episode_index": int(meta.get("episode_index", 0)),
            "start_time_sec": float(meta.get("start_time_sec", 0.0)),
            "end_time_sec": float(meta.get("end_time_sec", 0.0)),
            "episode_title": str(meta.get("episode_title", "")),
            "narrative_summary": str(meta.get("narrative_summary", "")),
            "actors": list(meta.get("actors", [])),
            "objects_detected": list(meta.get("objects_detected", [])),
            "audio_transcript_and_cues": str(meta.get("audio_transcript_and_cues", "")),
            "anomaly_or_security_flag": bool(meta.get("anomaly_or_security_flag", False)),
            "severity": str(meta.get("severity", "low")),
            "storyboard_rel_path": str(meta.get("storyboard_rel_path", "")),
            "audio_rel_path": str(meta.get("audio_rel_path", "")),
            "grid_used": str(meta.get("grid_used_str", "2x3")),
            "tile_actions": tile_actions,
        }

