# Manual Experiment Log Templates — Parts C & E

**Student:** Najart Rauf Awuni | **Index:** 10022300177
**IMPORTANT:** YOU MUST RUN THESE EXPERIMENTS YOURSELF and fill in REAL results.
Do NOT submit AI-generated numbers. The assessor will cross-check your outputs
against your deployed app. Fabricated logs will be treated as academic misconduct.

---

## PART C — Prompt Iteration Experiment Log

**Objective:** Compare output quality across the 3 prompt templates for the same query.

### Test Query 1

| Field | Value |
|-------|-------|
| Query | *(write the question you used)* |
| Dataset source retrieved | *(CSV / PDF / both)* |

| Template | Raw LLM Response | Hallucination? | Cites Source? | Answer Quality (1–5) | Notes |
|----------|-----------------|----------------|---------------|----------------------|-------|
| v1 (baseline) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | *(any observations)* |
| v2 (boundary + fallback) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | *(any observations)* |
| v3 (structured + citation) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | *(any observations)* |

**Analysis:** *(Write 3–5 sentences comparing the three outputs. What changed? Why?)*

---

### Test Query 2

| Field | Value |
|-------|-------|
| Query | *(write the question you used)* |
| Dataset source retrieved | *(CSV / PDF / both)* |

| Template | Raw LLM Response | Hallucination? | Cites Source? | Answer Quality (1–5) | Notes |
|----------|-----------------|----------------|---------------|----------------------|-------|
| v1 (baseline) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | |
| v2 (boundary + fallback) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | |
| v3 (structured + citation) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | |

**Analysis:** *(Write 3–5 sentences)*

---

### Test Query 3

| Field | Value |
|-------|-------|
| Query | *(write the question you used)* |
| Dataset source retrieved | *(CSV / PDF / both)* |

| Template | Raw LLM Response | Hallucination? | Cites Source? | Answer Quality (1–5) | Notes |
|----------|-----------------|----------------|---------------|----------------------|-------|
| v1 (baseline) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | |
| v2 (boundary + fallback) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | |
| v3 (structured + citation) | *(paste exact LLM output)* | Yes / No | Yes / No | _/5 | |

**Analysis:** *(Write 3–5 sentences)*

---

### Part C Summary Table

| Metric | V1 Baseline | V2 Boundary | V3 Structured |
|--------|------------|-------------|---------------|
| Avg hallucination rate | _% | _% | _% |
| Avg source citation rate | _% | _% | _% |
| Avg quality score (1–5) | _._ | _._ | _._ |
| Avg response length (words) | _ | _ | _ |

**Conclusion:** *(Which template performed best and why? 2–3 sentences)*

---

---

## PART E — Adversarial Query Experiment Log

**Objective:** Evaluate accuracy, hallucination, and consistency under adversarial conditions.
Compare RAG vs pure LLM (no context) for each query.

---

### Adversarial Query 1 — Ambiguous Query

**Query used:** *(e.g., "Who won?" — ambiguous: which election? which seat?)*

#### RAG Response (from your deployed app)
```
[Paste exact app response here]
```
**Retrieved chunk scores:**
- Chunk 1: source=___, score=___
- Chunk 2: source=___, score=___
- Chunk 3: source=___, score=___

#### Pure LLM Response (send SAME question with NO context — use the API directly)
```
[Paste exact LLM response here — NO context provided]
```

#### Evaluation

| Criterion | RAG Response | Pure LLM Response |
|-----------|-------------|-------------------|
| Factually accurate? | Yes / No / Partial | Yes / No / Partial |
| Hallucinated facts? | None / Minor / Major | None / Minor / Major |
| Consistent if asked twice? | Yes / No | Yes / No |
| Refused to guess? | Yes / No | Yes / No |
| Cited a source? | Yes / No | Yes / No |

**Analysis:** *(Evidence-based comparison — no opinions. Cite specific phrases from the responses.)*

---

### Adversarial Query 2 — Misleading / Presuppositional Query

**Query used:** *(e.g., "By how much did the NDC win the 2024 election?" — misleading: assumes NDC won)*

#### RAG Response (from your deployed app)
```
[Paste exact app response here]
```
**Retrieved chunk scores:**
- Chunk 1: source=___, score=___
- Chunk 2: source=___, score=___

#### Pure LLM Response (send SAME question with NO context)
```
[Paste exact LLM response here]
```

#### Evaluation

| Criterion | RAG Response | Pure LLM Response |
|-----------|-------------|-------------------|
| Detected false premise? | Yes / No | Yes / No |
| Hallucinated a winner? | Yes / No | Yes / No |
| Consistent if asked twice? | Yes / No | Yes / No |
| Refused to confirm false premise? | Yes / No | Yes / No |
| Cited a source? | Yes / No | Yes / No |

**Analysis:** *(Evidence-based comparison. Quote from both responses.)*

---

### Part E Summary: RAG vs Pure LLM Comparison

| Dimension | RAG System | Pure LLM (no context) |
|-----------|-----------|----------------------|
| Accuracy on in-domain questions (your tests) | _/10 correct | _/10 correct |
| Hallucination rate | _% | _% |
| Source grounding | Always cites | Never cites |
| Handles misleading queries | Correctly in _/2 cases | Correctly in _/2 cases |
| Response consistency (same Q twice) | High / Medium / Low | High / Medium / Low |

**Evidence-based conclusion:** *(3–5 sentences. Use your actual numbers. Do not write "I think" — write "The data shows…" or "In 2/2 adversarial cases, the RAG system…")*

---

## PART A — Chunking Strategy Comparison Log

**Objective:** Show the impact of chunking strategy on retrieval quality.

**Test query:** *(same query run with both strategies)*

| Metric | Fixed-size (512c / 64 overlap) | Sentence-boundary (~600c max) |
|--------|-------------------------------|-------------------------------|
| Number of chunks generated | ___ | ___ |
| Avg chunk length (chars) | ___ | ___ |
| Top-1 retrieved score | ___ | ___ |
| Top-5 avg score | ___ | ___ |
| Chunk contained complete sentence? | Yes _% / No _% | Yes _% / No _% |
| Response quality (1–5) | _ | _ |

**Analysis:** *(Which strategy worked better for this domain and why? 3–4 sentences)*
