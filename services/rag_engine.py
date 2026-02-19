"""
NyayaSathi AI — RAG Engine
============================
Production-ready Retrieval-Augmented Generation pipeline:
  1. **Chunk** reference documents into overlapping windows
  2. **Embed** chunks with SentenceTransformers
  3. **Index** embeddings in a FAISS flat-L2 index
  4. **Query** the index at inference time and return top-k chunks

The pipeline can also ingest the *uploaded* document so that the
AI Chat can answer questions grounded in the user's own file.

Public API:
  • build_index(documents)
  • save_index(path) / load_index(path)
  • query_index(query, top_k) → list of (chunk_text, score) dicts
  • get_context(query)       → formatted retrieval string
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from embeddings.embedding_model import EmbeddingModel
from utils.logger import get_logger
import config

logger = get_logger(__name__)


class RAGEngine:
    """FAISS-backed retrieval engine with text chunking."""

    def __init__(self, embedding_model: Optional[EmbeddingModel] = None) -> None:
        self._model = embedding_model or EmbeddingModel.get_instance()
        self._index: Optional[faiss.IndexFlatL2] = None
        self._chunks: list[str] = []         # parallel to FAISS vectors
        self._metadata: list[dict] = []      # optional doc-level metadata

    # ── Chunking ──────────────────────────────────────────────────────

    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = config.CHUNK_SIZE,
        overlap: int = config.CHUNK_OVERLAP,
    ) -> list[str]:
        """Split *text* into overlapping character windows.

        Returns
        -------
        list[str]
            Non-empty chunks.
        """
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - overlap
        return chunks

    # ── Index Management ──────────────────────────────────────────────

    def build_index(self, documents: list[str], metadata: Optional[list[dict]] = None) -> None:
        """Chunk, embed, and index a list of raw documents.

        Parameters
        ----------
        documents : list[str]
            Raw text of each document / section.
        metadata : list[dict] | None
            Optional per-document metadata (e.g. source filename).
        """
        all_chunks: list[str] = []
        all_meta: list[dict] = []

        for i, doc in enumerate(documents):
            doc_chunks = self.chunk_text(doc)
            all_chunks.extend(doc_chunks)
            doc_meta = (metadata[i] if metadata and i < len(metadata) else {})
            all_meta.extend([doc_meta] * len(doc_chunks))

        if not all_chunks:
            logger.warning("No chunks produced — index will be empty.")
            return

        logger.info("Embedding %d chunks …", len(all_chunks))
        embeddings = self._model.encode(all_chunks)          # (N, D)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatL2(dim)
        self._index.add(embeddings)                          # type: ignore[arg-type]
        self._chunks = all_chunks
        self._metadata = all_meta

        logger.info("FAISS index built — %d vectors, dim=%d", self._index.ntotal, dim)

    def save_index(self, directory: str = config.FAISS_INDEX_DIR) -> None:
        """Persist index + chunk texts to disk."""
        if self._index is None:
            raise RuntimeError("No index to save — call build_index first.")
        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self._index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump({"chunks": self._chunks, "metadata": self._metadata}, f, ensure_ascii=False)
        logger.info("Index saved to %s", directory)

    def load_index(self, directory: str = config.FAISS_INDEX_DIR) -> bool:
        """Load a previously saved index.  Returns True on success."""
        idx_path = os.path.join(directory, "index.faiss")
        chunks_path = os.path.join(directory, "chunks.json")

        if not os.path.isfile(idx_path) or not os.path.isfile(chunks_path):
            logger.warning("Index files not found at %s", directory)
            return False

        self._index = faiss.read_index(idx_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = data.get("chunks", [])
        self._metadata = data.get("metadata", [])
        logger.info(
            "Index loaded — %d vectors from %s",
            self._index.ntotal,
            directory,
        )
        return True

    # ── Querying ──────────────────────────────────────────────────────

    def query_index(
        self,
        query: str,
        top_k: int = config.TOP_K,
    ) -> list[dict]:
        """Retrieve the most relevant chunks for *query*.

        Returns
        -------
        list[dict]
            Each dict has keys ``text``, ``score`` (L2 distance — lower is
            better), and ``metadata``.
        """
        if self._index is None or self._index.ntotal == 0:
            logger.warning("Index is empty — returning no results.")
            return []

        query_emb = self._model.encode_single(query).reshape(1, -1)
        distances, indices = self._index.search(query_emb, min(top_k, self._index.ntotal))

        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            results.append({
                "text": self._chunks[idx],
                "score": float(dist),
                "metadata": self._metadata[idx] if idx < len(self._metadata) else {},
            })
        return results

    def get_context(self, query: str, top_k: int = config.TOP_K) -> str:
        """Return a formatted string of retrieved context for prompt injection."""
        results = self.query_index(query, top_k)
        if not results:
            return "No relevant context found."
        parts = []
        for i, r in enumerate(results, 1):
            parts.append(f"[Reference {i}]\n{r['text']}")
        return "\n\n".join(parts)

    # ── Convenience ───────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0

    def add_document(self, text: str, metadata: Optional[dict] = None) -> None:
        """Incrementally add a single document to an existing index."""
        chunks = self.chunk_text(text)
        if not chunks:
            return
        embeddings = self._model.encode(chunks)

        if self._index is None:
            self._index = faiss.IndexFlatL2(embeddings.shape[1])

        self._index.add(embeddings)  # type: ignore[arg-type]
        self._chunks.extend(chunks)
        self._metadata.extend([metadata or {}] * len(chunks))
        logger.info("Added %d chunks — total vectors: %d", len(chunks), self._index.ntotal)
