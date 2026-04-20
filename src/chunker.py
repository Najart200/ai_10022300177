"""
chunker.py — Two manual chunking strategies with no framework dependency.

Strategy A: Fixed-size character chunking with overlap
Strategy B: Sentence-boundary chunking (splits on '. ', '! ', '? ')

Why two strategies?
  - Fixed-size gives predictable, uniform chunks — good for dense structured
    data (CSV rows turned into sentences, budget tables).
  - Sentence-boundary preserves semantic units — better for narrative PDF text
    where breaking mid-sentence loses meaning.

Student: Najart Rauf Awuni | Index: 10022300177
"""

import re
import logging

logger = logging.getLogger(__name__)

# ── Tuneable defaults (justified in documentation) ──────────────────────────
FIXED_CHUNK_SIZE    = 512   # chars — fits well within MiniLM's 256-token limit
FIXED_CHUNK_OVERLAP = 64    # ~12.5% overlap — reduces boundary information loss
SENT_MAX_CHARS      = 600   # sentence chunks may merge up to this size
SENT_MIN_CHARS      = 80    # sentences shorter than this are merged with next


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY A — Fixed-size character chunking
# ─────────────────────────────────────────────────────────────────────────────

def fixed_size_chunk(text: str,
                     chunk_size: int = FIXED_CHUNK_SIZE,
                     overlap: int = FIXED_CHUNK_OVERLAP) -> list[str]:
    """
    Slide a window of `chunk_size` chars over `text`, stepping by
    (chunk_size - overlap) each time. Simple and deterministic.
    """
    if not text:
        return []

    chunks = []
    step   = chunk_size - overlap
    start  = 0

    while start < len(text):
        end   = start + chunk_size
        chunk = text[start:end].strip()
        if len(chunk) >= 20:            # skip trivially short tail chunks
            chunks.append(chunk)
        start += step

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  STRATEGY B — Sentence-boundary chunking
# ─────────────────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """
    Naive sentence splitter: split on '. ', '! ', '? ', '\n\n'.
    Avoids NLTK / spaCy dependency while preserving most sentence boundaries.
    """
    pattern = r"(?<=[.!?])\s+|(?<=\n)\n+"
    raw     = re.split(pattern, text)
    return [s.strip() for s in raw if s.strip()]


def sentence_chunk(text: str,
                   max_chars: int = SENT_MAX_CHARS,
                   min_chars: int = SENT_MIN_CHARS) -> list[str]:
    """
    Group sentences greedily: keep adding sentences to the current chunk
    until adding the next one would exceed `max_chars`. Very short sentences
    (< min_chars) are always merged with the next to avoid micro-chunks.
    """
    if not text:
        return []

    sentences = _split_sentences(text)
    chunks    = []
    current   = ""

    for sent in sentences:
        candidate = (current + " " + sent).strip() if current else sent

        if len(candidate) <= max_chars:
            current = candidate
        else:
            # current chunk is full — save it and start a new one
            if current and len(current) >= min_chars:
                chunks.append(current)
            current = sent

    if current and len(current) >= min_chars:
        chunks.append(current)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(docs: list[dict],
                    strategy: str = "sentence") -> list[dict]:
    """
    Chunk all documents using the chosen strategy.

    Args:
        docs:     output of data_loader.load_all_documents()
        strategy: "fixed" | "sentence"

    Returns:
        List of chunk dicts:
        {
          'text':     chunk text,
          'source':   original filename,
          'chunk_id': globally unique int,
          'strategy': strategy name,
          'page':     page number (PDF only, else None)
        }
    """
    assert strategy in ("fixed", "sentence"), \
        f"Unknown strategy '{strategy}'. Use 'fixed' or 'sentence'."

    all_chunks = []
    chunk_fn   = fixed_size_chunk if strategy == "fixed" else sentence_chunk
    cid        = 0

    for doc in docs:
        raw_text = doc.get("text", "")
        pieces   = chunk_fn(raw_text)

        for piece in pieces:
            all_chunks.append({
                "text":     piece,
                "source":   doc.get("source", "unknown"),
                "chunk_id": cid,
                "strategy": strategy,
                "page":     doc.get("page"),          # None for CSV rows
            })
            cid += 1

    logger.info(
        "Chunking (%s) complete — %d documents → %d chunks",
        strategy, len(docs), len(all_chunks)
    )
    return all_chunks
