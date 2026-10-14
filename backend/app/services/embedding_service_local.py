"""Local embedding service using sentence-transformers.

Replaces the HF Inference API path with a fully local model. No API key,
no quota, no rate limit. Model weights are loaded once into memory on
first use and cached.

Design:
    - Loads BAAI/bge-m3 (same model the API path used) so vectors are
      dimension-compatible (1024) with the existing Qdrant collection.
    - Batching on CPU is efficient: sentence-transformers pads to the
      longest sequence in the batch and runs one forward pass.
    - First call loads the model (~2.3 GB) — takes 20–60s depending on
      disk speed. Subsequent calls are sub-second per batch.
    - Runs the blocking encode() call inside a thread via asyncio.to_thread
      so the event loop stays responsive during HTTP requests.

Env vars used:
    EMBEDDING_MODEL      default: BAAI/bge-m3
    EMBEDDING_DIMENSION  default: 1024 (must match Qdrant collection)

Optionally set EMBEDDING_LOCAL_PATH to a local directory containing the
model (useful when the model was pre-downloaded, or when HF downloads
are unreliable). Otherwise the model is fetched from HF's model repo
(not the Inference API — so no quota is used).
"""

from __future__ import annotations

import asyncio
import logging
import os

from app.core.config import get_settings
from app.core.exceptions import RetrievalError

logger = logging.getLogger(__name__)


class LocalEmbeddingService:
    """Drop-in replacement for EmbeddingService using local inference."""

    _model = None  # class-level cache (one model per process)

    def __init__(self) -> None:
        settings = get_settings()
        self.dimension = settings.embedding_dimension
        self.model_name = settings.embedding_model
        self.local_path = os.getenv("EMBEDDING_LOCAL_PATH", "").strip() or None

    # ─── Model loading (lazy, cached) ───────────────────────────────
    @classmethod
    def _load_model(cls, model_ref: str):
        """Load and cache the SentenceTransformer model (blocking)."""
        from sentence_transformers import SentenceTransformer

        logger.info("Loading local embedding model: %s ...", model_ref)
        model = SentenceTransformer(model_ref)
        logger.info("Local embedding model loaded.")
        return model

    def _get_model(self):
        """Return the cached model, loading it on first access."""
        if self.__class__._model is None:
            source = self.local_path or self.model_name
            self.__class__._model = self._load_model(source)
        return self.__class__._model

    # ─── Public API (mirrors EmbeddingService) ──────────────────────
    async def embed(self, text: str) -> list[float]:
        """Generate a vector embedding for a single string."""
        try:
            vector = await asyncio.to_thread(self._encode_one, text)
            return vector
        except Exception as exc:
            logger.exception("Local embedding failed for single text.")
            raise RetrievalError(
                "Local embedding failed.",
                details={
                    "operation": "embed",
                    "text_length": len(text),
                    "error_type": type(exc).__name__,
                },
            ) from exc

    async def embed_many(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_retries: int = 1,
    ) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        batch_size is lower than the API path (64) because local CPU
        memory usage scales with batch size. 32 is a safe default for
        8 GB laptops.
        """
        if not texts:
            return []

        try:
            vectors = await asyncio.to_thread(
                self._encode_many, texts, batch_size
            )
            return vectors
        except Exception as exc:
            logger.exception("Local embedding failed for batch.")
            raise RetrievalError(
                "Local embedding failed for batch.",
                details={
                    "operation": "embed_many",
                    "batch_size": len(texts),
                    "error_type": type(exc).__name__,
                },
            ) from exc

    # ─── Blocking helpers (run in a thread) ─────────────────────────
    def _encode_one(self, text: str) -> list[float]:
        model = self._get_model()
        vec = model.encode(text, normalize_embeddings=True, convert_to_numpy=True)
        return [float(x) for x in vec.tolist()]

    def _encode_many(self, texts: list[str], batch_size: int) -> list[list[float]]:
        model = self._get_model()
        arr = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in row] for row in arr.tolist()]