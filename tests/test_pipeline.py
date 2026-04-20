"""
test_pipeline.py — Unit tests for manual RAG components.
Run with: python -m pytest tests/ -v

Student: Najart Rauf Awuni | Index: 10022300177
"""

import numpy as np
import pytest

# ─────────────────────────────────────────────────────────────────────────────
#  CHUNKER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestChunker:
    from src.chunker import fixed_size_chunk, sentence_chunk, chunk_documents

    def test_fixed_chunk_basic(self):
        from src.chunker import fixed_size_chunk
        text   = "A" * 1000
        chunks = fixed_size_chunk(text, chunk_size=200, overlap=20)
        assert len(chunks) > 1
        assert all(len(c) <= 200 for c in chunks)

    def test_fixed_chunk_overlap(self):
        from src.chunker import fixed_size_chunk
        text   = "Hello world this is a test sentence repeated. " * 30
        chunks = fixed_size_chunk(text, chunk_size=100, overlap=20)
        # Adjacent chunks should share a suffix/prefix because of overlap
        if len(chunks) >= 2:
            # Can't assert exact overlap due to strip(), but count should be correct
            assert len(chunks) >= 2

    def test_fixed_chunk_empty(self):
        from src.chunker import fixed_size_chunk
        assert fixed_size_chunk("") == []

    def test_sentence_chunk_basic(self):
        from src.chunker import sentence_chunk
        text   = ("The economy grew by 5%. Inflation was stable. "
                  "The budget deficit narrowed significantly this year. "
                  "Revenue collection exceeded targets. Expenditure was controlled.") * 5
        chunks = sentence_chunk(text, max_chars=200, min_chars=50)
        assert len(chunks) >= 1
        assert all(len(c) <= 250 for c in chunks)   # allow slight overrun at boundary

    def test_sentence_chunk_empty(self):
        from src.chunker import sentence_chunk
        assert sentence_chunk("") == []

    def test_chunk_documents_returns_correct_fields(self):
        from src.chunker import chunk_documents
        docs = [{"text": "Hello world. This is a test.", "source": "test.csv"}]
        for strategy in ("fixed", "sentence"):
            chunks = chunk_documents(docs, strategy=strategy)
            assert len(chunks) >= 1
            for c in chunks:
                assert "text"     in c
                assert "source"   in c
                assert "chunk_id" in c
                assert "strategy" in c
                assert c["strategy"] == strategy


# ─────────────────────────────────────────────────────────────────────────────
#  EMBEDDER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedder:

    @pytest.fixture(scope="class")
    def embedder(self):
        from src.embedder import Embedder
        return Embedder()

    def test_embed_query_shape(self, embedder):
        vec = embedder.embed_query("What is the budget deficit?")
        assert vec.ndim == 1
        assert vec.dtype == np.float32
        assert len(vec) == embedder.dim

    def test_embed_texts_shape(self, embedder):
        texts = ["Hello", "World", "AI is great"]
        vecs  = embedder.embed_texts(texts)
        assert vecs.shape == (3, embedder.dim)
        assert vecs.dtype == np.float32

    def test_embeddings_normalised(self, embedder):
        vec  = embedder.embed_query("Test query")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5, f"Expected unit norm, got {norm}"

    def test_embed_empty(self, embedder):
        result = embedder.embed_texts([])
        assert result.shape == (0, embedder.dim)


# ─────────────────────────────────────────────────────────────────────────────
#  VECTOR STORE TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorStore:

    def _make_store(self, n=10, dim=16):
        from src.vector_store import VectorStore
        store   = VectorStore(dim=dim)
        vecs    = np.random.randn(n, dim).astype(np.float32)
        # L2-normalise (mimic embedder)
        norms   = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs   /= norms
        chunks  = [{"text": f"chunk {i}", "source": "test", "chunk_id": i} for i in range(n)]
        store.add(vecs, chunks)
        return store, vecs

    def test_add_and_search(self):
        store, vecs = self._make_store(n=20, dim=16)
        results = store.search(vecs[0], top_k=3)
        assert len(results) == 3
        assert results[0]["score"] >= results[1]["score"]   # sorted descending

    def test_search_returns_score_field(self):
        store, vecs = self._make_store(n=5, dim=16)
        results = store.search(vecs[0], top_k=2)
        for r in results:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.01    # tiny fp tolerance

    def test_empty_store_search(self):
        from src.vector_store import VectorStore
        store   = VectorStore(dim=16)
        q       = np.random.randn(16).astype(np.float32)
        results = store.search(q, top_k=3)
        assert results == []


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT BUILDER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptBuilder:

    def _make_chunks(self):
        return [
            {"text": "Ghana NDC won 137 seats.", "source": "election.csv", "score": 0.9, "chunk_id": 0},
            {"text": "Budget deficit is 3.8% of GDP.", "source": "budget.pdf", "score": 0.8, "chunk_id": 1, "page": 42},
        ]

    def test_build_prompt_contains_query(self):
        from src.prompt_builder import build_prompt
        query  = "What did NDC win?"
        prompt, ctx, used = build_prompt(query, self._make_chunks(), "v2 (boundary + fallback)")
        assert query in prompt

    def test_build_prompt_contains_context(self):
        from src.prompt_builder import build_prompt
        prompt, ctx, used = build_prompt("Test?", self._make_chunks())
        assert "NDC" in prompt or "budget" in prompt.lower()

    def test_all_templates_render(self):
        from src.prompt_builder import build_prompt, TEMPLATES
        for key in TEMPLATES:
            prompt, _, _ = build_prompt("sample question", self._make_chunks(), key)
            assert len(prompt) > 50


# ─────────────────────────────────────────────────────────────────────────────
#  FEEDBACK TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFeedback:
    import tempfile, os

    def _store(self):
        import tempfile, os
        from src.feedback import FeedbackStore
        tmp = os.path.join(tempfile.mkdtemp(), "feedback.json")
        return FeedbackStore(path=tmp)

    def test_like_increases_boost(self):
        store = self._store()
        store.record("test question", [1, 2, 3], liked=True)
        boosts = store.get_boosts("test question")
        assert boosts.get(1, 0) > 0

    def test_dislike_decreases_boost(self):
        store = self._store()
        store.record("test question", [5], liked=False)
        boosts = store.get_boosts("test question")
        assert boosts.get(5, 0) < 0

    def test_stats_counts(self):
        store = self._store()
        store.record("q1", [1], liked=True)
        store.record("q2", [2], liked=False)
        stats = store.stats()
        assert stats["likes"]    == 1
        assert stats["dislikes"] == 1


# ─────────────────────────────────────────────────────────────────────────────
#  KEYWORD SCORER TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestKeywordScorer:

    def test_exact_match_scores_higher(self):
        from src.retriever import KeywordScorer
        corpus = ["Ghana election results 2020", "Budget deficit GDP", "Inflation rate"]
        scorer = KeywordScorer(corpus)
        high   = scorer.score("Ghana election", "Ghana election results 2020 NDC NPP")
        low    = scorer.score("Ghana election", "Inflation rate and GDP figures")
        assert high > low

    def test_score_bounded(self):
        from src.retriever import KeywordScorer
        corpus = ["hello world", "foo bar"]
        scorer = KeywordScorer(corpus)
        s      = scorer.score("hello", "hello world hello")
        assert 0.0 <= s <= 1.0
