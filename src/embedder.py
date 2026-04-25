"""
embedder.py — Custom embedding pipeline using sentence-transformers.
No LangChain / LlamaIndex. Embeddings are generated manually and stored
as a numpy array that FAISS can index directly.

Model choice: all-MiniLM-L6-v2
  - 384-dimensional dense vectors
  - Trained specifically for semantic similarity tasks
  - Runs on CPU in reasonable time (~800 sentences/sec on laptop)
  - Well-documented for RAG in academic papers

Student: Najart Rauf Awuni | Index: 10022300177
"""

import numpy as np
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Thin wrapper around SentenceTransformer for batch encoding."""

    def __init__(self, model_name: str = MODEL_NAME):
        logger.info("Loading embedding model: %s", model_name)
        # SentenceTransformer handles tokenization + pooling internally.
        # We don't need to call it differently — just .encode().
        self.model      = SentenceTransformer(model_name)
        self.model_name = model_name
        # sentence-transformers renamed this API; keep fallback for compatibility.
        if hasattr(self.model, "get_embedding_dimension"):
            self.dim = self.model.get_embedding_dimension()
        else:
            self.dim = self.model.get_sentence_embedding_dimension()
        logger.info("Embedding dimension: %d", self.dim)

    def embed_texts(self, texts: list[str],
                    batch_size: int = 64,
                    show_progress: bool = False) -> np.ndarray:
        """
        Encode a list of strings → float32 numpy array of shape (N, dim).

        batch_size=64 keeps memory usage reasonable on 8GB RAM laptops.
        normalize_embeddings=True makes cosine similarity = dot product,
        which is what FAISS IndexFlatIP computes (faster than L2 for cosine).
        """
        if not texts:
            return np.empty((0, self.dim), dtype=np.float32)

        vectors = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # unit vectors → cosine via dot product
            convert_to_numpy=True,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Encode a single query string → 1-D float32 array of length dim.
        Normalised the same way as corpus embeddings so scores are comparable.
        """
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec[0].astype(np.float32)
