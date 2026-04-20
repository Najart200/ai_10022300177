"""
vector_store.py — Manual FAISS vector store: build, save, load, search.
No framework wrappers — raw faiss-cpu calls only.

Index type: IndexFlatIP (exact inner-product / cosine search on unit vectors).
For production scale (>1M docs) you would switch to IndexIVFPQ, but for this
academic corpus (~3 000–5 000 chunks) exact search is fine and interpretable.

Student: Najart Rauf Awuni | Index: 10022300177
"""

import os
import json
import logging
import numpy as np
import faiss

logger = logging.getLogger(__name__)

INDEX_PATH    = "data/faiss.index"
METADATA_PATH = "data/metadata.json"


class VectorStore:
    """
    Wraps a FAISS flat index with a parallel metadata list so every
    vector position maps to its original chunk dict.
    """

    def __init__(self, dim: int):
        self.dim      = dim
        # IndexFlatIP: exact cosine search (works because embeddings are L2-normalised)
        self.index    = faiss.IndexFlatIP(dim)
        self.metadata: list[dict] = []   # parallel to index — index[i] ↔ metadata[i]

    # ── Build ────────────────────────────────────────────────────────────────

    def add(self, vectors: np.ndarray, chunks: list[dict]) -> None:
        """
        Add a batch of vectors + their corresponding chunk metadata.
        vectors shape: (N, dim), dtype float32.
        """
        assert vectors.shape[0] == len(chunks), \
            "Vector count must match chunk count."
        assert vectors.dtype == np.float32, \
            "FAISS requires float32 vectors."

        self.index.add(vectors)
        self.metadata.extend(chunks)
        logger.info("Added %d vectors. Index total: %d", len(chunks), self.index.ntotal)

    # ── Persist ──────────────────────────────────────────────────────────────

    def save(self, index_path: str = INDEX_PATH,
             meta_path: str = METADATA_PATH) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logger.info("Index saved to %s", index_path)

    @classmethod
    def load(cls, index_path: str = INDEX_PATH,
             meta_path: str = METADATA_PATH) -> "VectorStore":
        index    = faiss.read_index(index_path)
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        store          = cls.__new__(cls)
        store.dim      = index.d
        store.index    = index
        store.metadata = metadata
        logger.info("Index loaded — %d vectors", index.ntotal)
        return store

    @classmethod
    def exists(cls, index_path: str = INDEX_PATH,
               meta_path: str = METADATA_PATH) -> bool:
        return os.path.exists(index_path) and os.path.exists(meta_path)

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query_vec: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Return top-k results as list of dicts:
          {'text', 'source', 'chunk_id', 'score', ...original chunk fields}

        score ∈ [0, 1] — cosine similarity (higher = more relevant).
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty — cannot search.")
            return []

        # FAISS search expects shape (1, dim)
        q = query_vec.reshape(1, -1).astype(np.float32)
        scores, indices = self.index.search(q, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:          # FAISS returns -1 when fewer than k results exist
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        # Already sorted by score descending (FAISS guarantees this for IP)
        return results
