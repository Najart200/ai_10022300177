"""
prompt_builder.py — Manual prompt construction for the RAG pipeline.

Implements three prompt iterations (Part C) selectable at runtime.
Context window management: hard-cap total context at ~3 000 chars to stay
within the LLM's context window while leaving room for the model's response.

Hallucination control techniques used:
  1. "Based ONLY on the context below" instruction
  2. "If the answer is not in the context, say 'I don't have that information.'"
  3. Source citation instruction ("cite the source you used")

Student: Najart Rauf Awuni | Index: 10022300177
"""

import logging

logger = logging.getLogger(__name__)

MAX_CONTEXT_CHARS = 3_000   # hard cap to fit inside most free-tier LLMs


# ─────────────────────────────────────────────────────────────────────────────
#  PROMPT TEMPLATES  (3 iterations — Part C)
# ─────────────────────────────────────────────────────────────────────────────

# Iteration 1 — Baseline (too open, led to hallucination in testing)
TEMPLATE_V1 = """\
You are a helpful assistant. Use the provided context to answer the question.

Context:
{context}

Question: {question}
Answer:"""


# Iteration 2 — Added explicit boundary + no-info fallback
TEMPLATE_V2 = """\
You are a knowledgeable assistant for Academic City University.
Answer ONLY based on the context below. If the answer is not in the context,
respond with: "I don't have that information in my knowledge base."

Context:
{context}

Question: {question}
Answer:"""


# Iteration 3 — Best version: structured, cites source, handles ambiguity
TEMPLATE_V3 = """\
You are ACity Scholar, an AI research assistant for Academic City University, Ghana.
Your knowledge comes EXCLUSIVELY from the documents below.

RULES:
1. Answer concisely and factually using ONLY the context provided.
2. If the context does not contain enough information, say:
   "The provided documents do not contain sufficient information to answer this."
3. Do NOT invent statistics, names, dates, or policy details.
4. After your answer, list the source(s) you used in this format:
   Source: [filename, page/row if available]

--- CONTEXT START ---
{context}
--- CONTEXT END ---

User Question: {question}

ACity Scholar Response:"""


TEMPLATES = {
    "v1 (baseline)":          TEMPLATE_V1,
    "v2 (boundary + fallback)": TEMPLATE_V2,
    "v3 (structured + citation)": TEMPLATE_V3,
}

DEFAULT_TEMPLATE = "v3 (structured + citation)"


# ─────────────────────────────────────────────────────────────────────────────
#  CONTEXT WINDOW MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

def _build_context_block(chunks: list[dict],
                         max_chars: int = MAX_CONTEXT_CHARS) -> tuple[str, list[dict]]:
    """
    Greedily add chunks (highest score first) until we hit max_chars.
    Returns (context_string, chunks_actually_used).
    """
    used   = []
    parts  = []
    budget = max_chars

    for chunk in chunks:            # already sorted by score descending
        snippet = (
            f"[Source: {chunk.get('source', 'unknown')}"
            + (f", Page {chunk['page']}" if chunk.get("page") else "")
            + f" | Score: {chunk.get('score', 0):.4f}]\n"
            + chunk["text"]
        )
        if len(snippet) <= budget:
            parts.append(snippet)
            used.append(chunk)
            budget -= len(snippet) + 2   # +2 for "\n\n" separator
        else:
            # Trim the snippet to remaining budget
            trimmed = snippet[:budget]
            if len(trimmed) > 60:
                parts.append(trimmed + "...")
                used.append(chunk)
            break

    context = "\n\n".join(parts)
    logger.debug("Context block: %d chars from %d chunks", len(context), len(used))
    return context, used


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def build_prompt(query: str,
                 chunks: list[dict],
                 template_key: str = DEFAULT_TEMPLATE) -> tuple[str, str, list[dict]]:
    """
    Build the final prompt string.

    Returns:
        (prompt_string, context_block, chunks_used)
    """
    template = TEMPLATES.get(template_key, TEMPLATES[DEFAULT_TEMPLATE])
    context, used = _build_context_block(chunks)

    prompt = template.format(context=context, question=query)
    logger.info(
        "Prompt built — template=%s | context=%d chars | chunks=%d",
        template_key, len(context), len(used)
    )
    return prompt, context, used
