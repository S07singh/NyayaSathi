"""
NyayaSathi AI — Embedding Model Wrapper
========================================
Thin wrapper around SentenceTransformers that provides:
- Singleton-style model caching (avoids reloading on every call)
- Batch and single-text encoding helpers
- Dimension introspection for FAISS index creation

Architecture note:
  We keep embedding logic isolated so that swapping to a different
  model (e.g. BGE, Instructor) requires changes in only ONE place.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

import config

logger = get_logger(__name__)

# Module-level cache — one model instance per model name.
_MODEL_CACHE: dict[str, "EmbeddingModel"] = {}


class EmbeddingModel:
    """Reusable embedding encoder backed by SentenceTransformers."""

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL_NAME
        logger.info("Loading embedding model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name)
        self.dimension: int = self._model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dimension)

    # ── Public API ────────────────────────────────────────────────────

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Encode a list of texts → (N, D) float32 numpy array."""
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text → (D,) float32 numpy array."""
        return self.encode([text])[0]

    # ── Factory (cached) ──────────────────────────────────────────────

    @classmethod
    def get_instance(cls, model_name: str | None = None) -> "EmbeddingModel":
        """Return a cached model instance to avoid repeated loads."""
        key = model_name or config.EMBEDDING_MODEL_NAME
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = cls(model_name=key)
        return _MODEL_CACHE[key]
