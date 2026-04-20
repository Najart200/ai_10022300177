"""
retriever.py — Top-k retrieval with HYBRID SEARCH extension (Part B).

Extension chosen: Hybrid Search
  Combines dense semantic similarity (FAISS cosine) with a manual sparse
  BM25-inspired keyword score computed with pure numpy/string ops.
  No external BM25 library used — the IDF and TF are computed from scratch.

Why hybrid?
  Pure dense search failed on exact queries like "NDC seats 2020" because the
  embedding model maps them to a broad political-party concept rather than
  the exact number. Hybrid search boosts chunks that literally contain the
  keywords, fixing this failure mode.

Failure cases documented in docs/full_documentation.md (Part B section).

Student: Najart Rauf Awuni | Index: 10022300177
"""

import math
import logging
import numpy as np
from src.vector_store import VectorStore
from src.embedder import Embedder

logger = logging.getLogger(__name__)

# Hybrid weight: α * dense + (1-α) * sparse
ALPHA = 0.70   # dense dominates; tweak for domain shift


# ─────────────────────────────────────────────────────────────────────────────
#  SPARSE (keyword) SCORING — manual BM25-lite
# ─────────────────────────────────────────────────────────────────────────────

class KeywordScorer:
    """
    Lightweight IDF-weighted keyword scorer built from the chunk corpus.
    Uses token overlap (word frequency) without any external NLP library.
    """

    def __init__(self, corpus_texts: list[str]):
        self._doc_count = len(corpus_texts)
        self._idf       = self._build_idf(corpus_texts)

    @staticmethod
    def _tokenise(text: str) -> list[str]:
        return text.lower().split()

    def _build_idf(self, corpus: list[str]) -> dict[str, float]:
        """IDF = log( (N+1) / (df+1) ) — smoothed to avoid division by zero."""
        df: dict[str, int] = {}
        for text in corpus:
            seen = set(self._tokenise(text))
            for token in seen:
                df[token] = df.get(token, 0) + 1

        n = self._doc_count
        return {
            tok: math.log((n + 1) / (count + 1)) + 1.0
            for tok, count in df.items()
        }

    def score(self, query: str, text: str) -> float:
        """
        TF-IDF dot-product similarity between query terms and chunk text.
        Normalised to [0, 1] range by dividing by maximum possible score.
        """
        q_tokens = self._tokenise(query)
        t_tokens = self._tokenise(text)

        if not q_tokens or not t_tokens:
            return 0.0

        t_freq: dict[str, int] = {}
        for tok in t_tokens:
            t_freq[tok] = t_freq.get(tok, 0) + 1

        raw = sum(
            self._idf.get(tok, 0.0) * t_freq.get(tok, 0)
            for tok in set(q_tokens)
        )
        # Normalise: divide by (query_idf_sum * max_term_freq) to bound [0,1]
        denom = sum(self._idf.get(tok, 0.0) for tok in set(q_tokens)) * max(t_freq.values(), default=1)
        return raw / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  RETRIEVER
# ─────────────────────────────────────────────────────────────────────────────

class Retriever:
    """
    Combines VectorStore (dense) and KeywordScorer (sparse) into one
    hybrid retrieval interface.
    """

    def __init__(self, store: VectorStore, embedder: Embedder,
                 alpha: float = ALPHA):
        self.store    = store
        self.embedder = embedder
        self.alpha    = alpha

        # Build keyword scorer from all stored chunk texts
        corpus = [m["text"] for m in store.metadata]
        self.kw_scorer = KeywordScorer(corpus)
        logger.info("Retriever initialised — %d chunks in index", len(corpus))

    def retrieve(self, query: str,
                 top_k: int = 5,
                 use_hybrid: bool = True,
                 feedback_boosts: dict | None = None) -> list[dict]:
        """
        Retrieve top-k chunks for a query.

        Args:
            query:           natural language question
            top_k:           number of final results
            use_hybrid:      True → hybrid dense+sparse; False → dense only
            feedback_boosts: {chunk_id: boost_delta} from feedback module (Part G)

        Returns:
            Sorted list of chunk dicts with 'score', 'dense_score',
            'sparse_score' fields.
        """
        # --- Dense retrieval (over-fetch to allow re-ranking) ----------------
        fetch_k     = min(top_k * 4, self.store.index.ntotal)
        query_vec   = self.embedder.embed_query(query)
        dense_hits  = self.store.search(query_vec, top_k=fetch_k)

        if not dense_hits:
            return []

        if not use_hybrid:
            # Pure dense — just trim and return
            for h in dense_hits:
                h["dense_score"]  = h["score"]
                h["sparse_score"] = 0.0
            return dense_hits[:top_k]

        # --- Hybrid re-scoring -----------------------------------------------
        for hit in dense_hits:
            dense_s  = hit["score"]                          # already cosine [0,1]
            sparse_s = self.kw_scorer.score(query, hit["text"])
            combined = self.alpha * dense_s + (1 - self.alpha) * sparse_s

            # Part G: apply per-chunk feedback boost (positive feedback → +0.05)
            if feedback_boosts and hit.get("chunk_id") in feedback_boosts:
                combined += feedback_boosts[hit["chunk_id"]]

            hit["dense_score"]  = round(dense_s,  4)
            hit["sparse_score"] = round(sparse_s, 4)
            hit["score"]        = round(combined, 4)

        # Re-rank by hybrid score
        dense_hits.sort(key=lambda x: x["score"], reverse=True)
        logger.info(
            "Hybrid retrieval — query: '%s' | top score: %.4f",
            query[:60], dense_hits[0]["score"] if dense_hits else 0
        )
        return dense_hits[:top_k]
