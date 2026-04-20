"""
pipeline.py — Full RAG pipeline with logging at every stage (Part D).

Stage order:
  1. Query received
  2. Embed query
  3. Hybrid retrieval (dense + sparse)
  4. Context selection / context window management
  5. Prompt construction
  6. LLM call
  7. Response returned

Every stage logs its inputs, outputs, and timing.

Student: Najart Rauf Awuni | Index: 10022300177
"""

import time
import logging
import json
import os
from datetime import datetime

from src.data_loader import load_all_documents
from src.chunker     import chunk_documents
from src.embedder    import Embedder
from src.vector_store import VectorStore
from src.retriever   import Retriever
from src.prompt_builder import build_prompt, TEMPLATES, DEFAULT_TEMPLATE
from src.llm_client  import call_llm
from src.feedback    import FeedbackStore

logger = logging.getLogger(__name__)

LOG_DIR  = "logs"
os.makedirs(LOG_DIR, exist_ok=True)


def _ts() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
#  INDEX BUILDING  (run once; persisted to disk)
# ─────────────────────────────────────────────────────────────────────────────

def build_index(strategy: str = "sentence",
                force_rebuild: bool = False) -> tuple[VectorStore, Embedder]:
    """
    Download data → chunk → embed → store in FAISS.
    Skips rebuild if index already exists (unless force_rebuild=True).
    """
    embedder = Embedder()
    store    = VectorStore(dim=embedder.dim)

    if VectorStore.exists() and not force_rebuild:
        logger.info("Loading existing FAISS index from disk.")
        store = VectorStore.load()
        return store, embedder

    logger.info("=== BUILD INDEX — strategy: %s ===", strategy)

    # Stage 1: Load documents
    t0   = time.time()
    docs = load_all_documents()
    logger.info("Stage 1 LOAD: %d documents in %.2fs", len(docs), time.time() - t0)

    # Stage 2: Chunk
    t0     = time.time()
    chunks = chunk_documents(docs, strategy=strategy)
    logger.info("Stage 2 CHUNK: %d chunks in %.2fs", len(chunks), time.time() - t0)

    # Stage 3: Embed
    t0      = time.time()
    texts   = [c["text"] for c in chunks]
    vectors = embedder.embed_texts(texts, show_progress=True)
    logger.info("Stage 3 EMBED: shape=%s in %.2fs", vectors.shape, time.time() - t0)

    # Stage 4: Store
    t0 = time.time()
    store.add(vectors, chunks)
    store.save()
    logger.info("Stage 4 STORE: saved in %.2fs", time.time() - t0)

    return store, embedder


# ─────────────────────────────────────────────────────────────────────────────
#  QUERY PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Stateful pipeline that holds a built index and processes queries.
    """

    def __init__(self, strategy: str = "sentence",
                 force_rebuild: bool = False):
        self.store, self.embedder = build_index(strategy, force_rebuild)
        self.retriever  = Retriever(self.store, self.embedder)
        self.feedback   = FeedbackStore()
        self.history: list[dict] = []        # conversation memory (Part G)

    def query(self, question: str,
              top_k: int = 5,
              template_key: str = DEFAULT_TEMPLATE,
              use_hybrid: bool = True,
              temperature: float = 0.2) -> dict:
        """
        Run the full RAG pipeline for one question.
        Returns a rich dict with every intermediate artefact for UI display.
        """
        run_log: dict = {
            "timestamp":    _ts(),
            "question":     question,
            "template":     template_key,
            "use_hybrid":   use_hybrid,
            "top_k":        top_k,
        }

        # ── Stage 1: Embed query ──────────────────────────────────────────
        t0        = time.time()
        query_vec = self.embedder.embed_query(question)
        run_log["stage1_embed_ms"] = round((time.time() - t0) * 1000, 1)
        logger.info("[Stage 1] Query embedded in %.1f ms", run_log["stage1_embed_ms"])

        # ── Stage 2: Hybrid retrieval ──────────────────────────────────────
        t0     = time.time()
        boosts = self.feedback.get_boosts(question)
        chunks = self.retriever.retrieve(
            question, top_k=top_k,
            use_hybrid=use_hybrid,
            feedback_boosts=boosts,
        )
        run_log["stage2_retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        run_log["retrieved_chunks"]    = chunks
        logger.info(
            "[Stage 2] Retrieved %d chunks in %.1f ms | top score=%.4f",
            len(chunks), run_log["stage2_retrieval_ms"],
            chunks[0]["score"] if chunks else 0,
        )

        # ── Stage 3: Context selection & prompt construction ──────────────
        t0            = time.time()
        # Inject conversation history for memory (Part G)
        history_text  = self._format_history()
        augmented_q   = (history_text + "\n\nCurrent Question: " + question
                         if history_text else question)

        prompt, context_block, used_chunks = build_prompt(
            augmented_q, chunks, template_key
        )
        run_log["stage3_prompt_ms"] = round((time.time() - t0) * 1000, 1)
        run_log["prompt"]           = prompt
        run_log["context_block"]    = context_block
        run_log["chunks_used"]      = used_chunks
        logger.info(
            "[Stage 3] Prompt built in %.1f ms | %d chars | %d chunks used",
            run_log["stage3_prompt_ms"], len(prompt), len(used_chunks)
        )

        # ── Stage 4: LLM call ─────────────────────────────────────────────
        t0      = time.time()
        llm_out = call_llm(prompt, temperature=temperature)
        run_log["stage4_llm_ms"]  = round((time.time() - t0) * 1000, 1)
        run_log["llm_result"]     = llm_out
        run_log["response"]       = llm_out["response"]
        logger.info(
            "[Stage 4] LLM responded in %.1f ms | provider=%s | tokens_out=%d",
            run_log["stage4_llm_ms"], llm_out.get("provider"), llm_out.get("tokens_out")
        )

        # ── Stage 5: Update conversation history ──────────────────────────
        self.history.append({
            "question": question,
            "answer":   llm_out["response"][:300],   # truncate for memory
        })
        if len(self.history) > 5:    # keep last 5 turns only
            self.history.pop(0)

        # ── Persist log ───────────────────────────────────────────────────
        self._write_log(run_log)

        return run_log

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _format_history(self) -> str:
        if not self.history:
            return ""
        turns = []
        for turn in self.history[-3:]:   # last 3 turns for context
            turns.append(f"Q: {turn['question']}\nA: {turn['answer']}")
        return "Previous conversation:\n" + "\n---\n".join(turns)

    def _write_log(self, run_log: dict) -> None:
        log_path = os.path.join(LOG_DIR, f"query_{_ts().replace(':', '-')}.json")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                # Serialise — convert numpy values to python types
                safe = json.loads(json.dumps(run_log, default=str))
                json.dump(safe, f, indent=2)
        except Exception as e:
            logger.warning("Could not write query log: %s", e)
