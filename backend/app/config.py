"""Configuration module for SieveRAG.

Loads environment variables and defines application, storage,
and vlm-sieve ingestion hyperparameters.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from vlmsieve import SieveConfig

# Load environment variables from .env file if present
load_dotenv()


class Settings(BaseModel):
    """Central configuration settings for SieveRAG backend."""

    # Application Info
    APP_NAME: str = Field(default="SieveRAG Backend", description="Name of the service")
    APP_VERSION: str = Field(default="0.1.0", description="Semantic service version")
    DEBUG: bool = Field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1", "t"),
        description="Debug mode toggle",
    )
    HOST: str = Field(
        default_factory=lambda: os.getenv("HOST", "0.0.0.0"),
        description="API bind host",
    )
    PORT: int = Field(
        default_factory=lambda: int(os.getenv("PORT", "8000")),
        description="API bind port",
    )

    # Base & Storage Directories
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent,
        description="Root backend directory",
    )
    DATA_DIR: Path = Field(
        default_factory=lambda: Path(
            os.getenv("DATA_DIR", str(Path(__file__).resolve().parent.parent / "data"))
        ),
        description="Root data directory for persistent video RAG assets",
    )

    # Derived Storage Paths
    @property
    def STORAGE_DIR(self) -> Path:
        return self.DATA_DIR / "storage"

    @property
    def UPLOADS_DIR(self) -> Path:
        return self.STORAGE_DIR / "uploads"

    @property
    def STORYBOARDS_DIR(self) -> Path:
        return self.STORAGE_DIR / "storyboards"

    @property
    def AUDIO_DIR(self) -> Path:
        return self.STORAGE_DIR / "audio"

    # vlm-sieve Ingestion Parameters (Phase 1 Baseline)
    SIEVE_SAMPLE_FPS: float = Field(
        default_factory=lambda: float(os.getenv("SIEVE_SAMPLE_FPS", "10.0")),
        description="Frame rate at which raw video is sampled by the motion gatekeeper",
    )
    SIEVE_PRE_ROLL_SEC: float = Field(
        default_factory=lambda: float(os.getenv("SIEVE_PRE_ROLL_SEC", "3.5")),
        description="Lookback window in seconds preserved prior to detected motion onset",
    )
    SIEVE_MERGE_GAP_SEC: float = Field(
        default_factory=lambda: float(os.getenv("SIEVE_MERGE_GAP_SEC", "6.0")),
        description="Maximum temporal gap in seconds between motion bursts to merge into one episode",
    )
    SIEVE_TILE_MAX_DIM: int = Field(
        default_factory=lambda: int(os.getenv("SIEVE_TILE_MAX_DIM", "640")),
        description="Maximum pixel dimension (width/height) for each tile in composite storyboard",
    )
    SIEVE_SPATIAL_GRID_OVERLAY: bool = Field(
        default_factory=lambda: os.getenv("SIEVE_SPATIAL_GRID_OVERLAY", "true").lower()
        in ("true", "1", "t"),
        description="Whether to render 4x4 coordinate grid cells (A1..D4) on storyboard tiles",
    )
    SIEVE_BURN_TIMESTAMPS: bool = Field(
        default_factory=lambda: os.getenv("SIEVE_BURN_TIMESTAMPS", "true").lower()
        in ("true", "1", "t"),
        description="Whether to burn chronological timestamp badge (#N | +X.XXs) into tiles",
    )
    SIEVE_SUPPRESS_DISTURBANCES: bool = Field(
        default_factory=lambda: os.getenv("SIEVE_SUPPRESS_DISTURBANCES", "true").lower()
        in ("true", "1", "t"),
        description="Whether to suppress global camera shake / exposure flashes with no localized foreground motion",
    )
    SIEVE_MOTION_THRESHOLD: float = Field(
        default_factory=lambda: float(os.getenv("SIEVE_MOTION_THRESHOLD", "0.015")),
        description="Normalized pixel delta threshold to trigger motion detection",
    )
    SIEVE_TOKENS_PER_IMAGE: int = Field(
        default_factory=lambda: int(os.getenv("SIEVE_TOKENS_PER_IMAGE", "1120")),
        description="Nominal input token cost of one composite storyboard image in frontier VLMs",
    )

    # Direct Video Baseline Token Rate
    DIRECT_VIDEO_TOKENS_PER_SEC: float = Field(
        default_factory=lambda: float(os.getenv("DIRECT_VIDEO_TOKENS_PER_SEC", "91.5")),
        description="Empirical token consumption rate for direct raw video ingestion (tokens/sec)",
    )

    # Audio Slicing Hyperparameters
    AUDIO_SAMPLE_RATE: int = Field(
        default_factory=lambda: int(os.getenv("AUDIO_SAMPLE_RATE", "16000")),
        description="Sample rate for sliced audio (Hz, optimized for Whisper / Gemini Speech)",
    )
    AUDIO_CHANNELS: int = Field(
        default=1,
        description="Number of audio channels (1 = mono)",
    )
    AUDIO_BITRATE: str = Field(
        default="32k",
        description="Audio encoding bitrate for lightweight MP3 storage",
    )
    AUDIO_PAD_SEC: float = Field(
        default_factory=lambda: float(os.getenv("AUDIO_PAD_SEC", "0.5")),
        description="Padding in seconds applied to audio slice boundaries [start - pad, end + pad]",
    )

    # Gemini & Pinecone Credentials & Hyperparameters (Phase 2)
    GEMINI_API_KEY: str = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", ""),
        description="Google Gemini API key for multimodal reasoning and indexing",
    )
    GEMINI_MODEL: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", os.getenv("GEMINI_VLM_MODEL", "gemini-3.8-flash")),
        description="Target Gemini model for video episode extraction",
    )
    GEMINI_VLM_MODEL: str = Field(
        default_factory=lambda: os.getenv("GEMINI_VLM_MODEL", "gemini-3.8-flash"),
        description="Target Gemini VLM model for video episode extraction",
    )
    GEMINI_EMBEDDING_MODEL: str = Field(
        default_factory=lambda: os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"),
        description="Target Gemini embedding model for semantic retrieval",
    )
    EMBEDDING_DIMENSION: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "768")),
        description="Dimensionality of text embedding vectors (default 768 for text-embedding-004)",
    )
    PINECONE_API_KEY: str = Field(
        default_factory=lambda: os.getenv("PINECONE_API_KEY", ""),
        description="Pinecone API key for vector storage",
    )
    PINECONE_INDEX_NAME: str = Field(
        default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "sieverag-episodes"),
        description="Pinecone vector index name for episode embeddings",
    )
    PINECONE_NAMESPACE: str = Field(
        default_factory=lambda: os.getenv("PINECONE_NAMESPACE", "default"),
        description="Pinecone index namespace for multi-tenant isolation",
    )
    DEFAULT_TARGET_VLM: str = Field(
        default_factory=lambda: os.getenv("DEFAULT_TARGET_VLM", "gemini"),
        description="Default target VLM architecture profile for vlm-sieve (gemini | gpt-4o | qwen-vl | local-7b)",
    )

    def ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        for path in (self.DATA_DIR, self.STORAGE_DIR, self.UPLOADS_DIR, self.STORYBOARDS_DIR, self.AUDIO_DIR):
            path.mkdir(parents=True, exist_ok=True)

    def to_sieve_config(self, **overrides: Any) -> SieveConfig:
        """Construct a validated vlmsieve.SieveConfig instance from settings.

        Args:
            **overrides: Optional runtime keyword overrides for SieveConfig fields.

        Returns:
            Configured SieveConfig instance.
        """
        config_data: dict[str, Any] = {
            "sample_fps": self.SIEVE_SAMPLE_FPS,
            "pre_roll_sec": self.SIEVE_PRE_ROLL_SEC,
            "merge_gap_sec": self.SIEVE_MERGE_GAP_SEC,
            "tile_max_dim": self.SIEVE_TILE_MAX_DIM,
            "spatial_grid_overlay": self.SIEVE_SPATIAL_GRID_OVERLAY,
            "burn_timestamps": self.SIEVE_BURN_TIMESTAMPS,
            "suppress_disturbances": self.SIEVE_SUPPRESS_DISTURBANCES,
            "motion_threshold": self.SIEVE_MOTION_THRESHOLD,
            "tokens_per_image": self.SIEVE_TOKENS_PER_IMAGE,
        }
        config_data.update(overrides)
        return SieveConfig(**config_data)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton instance of application Settings."""
    settings = Settings()
    settings.ensure_directories()
    return settings

