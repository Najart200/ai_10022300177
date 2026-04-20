"""
feedback.py — Part G: Simple user feedback loop for retrieval improvement.

When the user gives a thumbs-up on a response, we record which chunk_ids
were used and apply a small score boost the next time similar queries arrive.
When they give thumbs-down, we apply a penalty.

The boost is stored in a JSON file so it persists across sessions.
This is NOT a full learning-to-rank system — it is a lightweight, transparent
heuristic that demonstrably changes retrieval rankings and is easy to inspect.

Innovation justification:
  Standard RAG systems have no feedback mechanism. This module closes the loop
  so the system improves from each interaction without retraining the embedder.

Student: Najart Rauf Awuni | Index: 10022300177
"""

import os
import json
import logging

logger = logging.getLogger(__name__)

FEEDBACK_PATH   = "data/feedback.json"
BOOST_ON_LIKE   = +0.05
BOOST_ON_DISLIKE = -0.04


class FeedbackStore:
    """
    Persists a mapping: chunk_id (str) → cumulative score delta.
    Also stores (query_hash → chunk_id list) so we can apply boosts to
    semantically similar future queries.
    """

    def __init__(self, path: str = FEEDBACK_PATH):
        self.path = path
        self._data: dict = self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"boosts": {}, "history": []}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ── Record feedback ───────────────────────────────────────────────────────

    def record(self, question: str,
               chunk_ids: list[int],
               liked: bool) -> None:
        """
        Record user feedback for the chunks used in answering `question`.

        Args:
            question:  the original query
            chunk_ids: list of chunk_id ints from retrieved chunks
            liked:     True = thumbs up, False = thumbs down
        """
        delta = BOOST_ON_LIKE if liked else BOOST_ON_DISLIKE
        boosts = self._data["boosts"]

        for cid in chunk_ids:
            key        = str(cid)
            boosts[key] = round(boosts.get(key, 0.0) + delta, 4)

        self._data["history"].append({
            "question":  question[:120],
            "chunk_ids": chunk_ids,
            "liked":     liked,
        })
        # Keep only last 200 entries
        self._data["history"] = self._data["history"][-200:]
        self._save()

        logger.info(
            "Feedback recorded: %s for %d chunks (delta=%.2f)",
            "LIKED" if liked else "DISLIKED", len(chunk_ids), delta
        )

    # ── Retrieve boosts ───────────────────────────────────────────────────────

    def get_boosts(self, question: str) -> dict[int, float]:
        """
        Return {chunk_id (int) → score_delta} for use in retriever re-ranking.
        Currently returns all recorded boosts regardless of query similarity
        (simple implementation — a production system would filter by query).
        """
        raw = self._data.get("boosts", {})
        return {int(k): v for k, v in raw.items() if abs(v) > 0.001}

    # ── Stats for UI ──────────────────────────────────────────────────────────

    def stats(self) -> dict:
        boosts   = self._data.get("boosts", {})
        history  = self._data.get("history", [])
        likes    = sum(1 for h in history if h["liked"])
        dislikes = sum(1 for h in history if not h["liked"])
        return {
            "total_feedback": len(history),
            "likes":          likes,
            "dislikes":       dislikes,
            "chunks_boosted": sum(1 for v in boosts.values() if v > 0),
            "chunks_penalised": sum(1 for v in boosts.values() if v < 0),
        }
