#  ACity Scholar — RAG Chatbot

> **CS4241 Introduction to Artificial Intelligence — 2026**
> Student: **Najart Rauf Awuni** | Index: **10022300177**

A fully custom Retrieval-Augmented Generation (RAG) chatbot for Academic City University College, built **without LangChain, LlamaIndex, Haystack, or any pre-built RAG framework**. Every component — chunking, embedding, vector storage, hybrid retrieval, prompt construction — is implemented manually from scratch.

---

      Repository Structure

```
ai_10022300177/
├── app.py                          # Streamlit UI (purple/white panda theme)
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/
│   ├── config.toml                 # Purple theme config
│   └── secrets.toml                # API keys (not committed)
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Downloads + cleans CSV & PDF
│   ├── chunker.py                  # Fixed-size & sentence-boundary chunking
│   ├── embedder.py                 # SentenceTransformer embedding pipeline
│   ├── vector_store.py             # FAISS IndexFlatIP (manual)
│   ├── retriever.py                # Hybrid search (dense + manual BM25-lite)
│   ├── prompt_builder.py           # 3 prompt template iterations
│   ├── llm_client.py               # Groq API via raw requests (no SDK)
│   ├── pipeline.py                 # Full pipeline + per-stage logging
│   ├── feedback.py                 # Feedback loop (Part G novel feature)
│   └── utils.py                    # Logging setup, UI helpers
├── assets/
│   └── panda.svg                   # Animated thinking panda mascot
├── data/
│   └── .gitkeep                    # Index + feedback stored here at runtime
├── logs/
│   └── .gitkeep                    # Per-query JSON logs (Part D)
├── docs/
│   ├── architecture.md             # Mermaid diagram + Part F justification
│   ├── experiment_log_template.md  # Manual experiment templates (Parts C, E, A)
│   └── full_documentation.md       # Complete Parts A–G documentation
└── tests/
    └── test_pipeline.py            # Unit tests for all manual components
```

---

##  Knowledge Base

| Dataset | Format | Description |
|---------|--------|-------------|
| Ghana Election Results | CSV | Constituency-level results — parties, seats, vote counts |
| 2025 Ghana Budget Statement | PDF | Fiscal policy, revenue targets, sector spending |

---

##  How It Works

```
Query → Embed → Hybrid Retrieve (Dense FAISS + Manual TF-IDF) →
Context Selection → Prompt (Template V3) → Groq LLM → Response
                                                    ↕
                            Feedback Loop (👍/👎 → chunk boost)
```

### RAG Components (all manual — no framework)
| Component | Implementation |
|-----------|---------------|
| Chunking | `chunker.py` — fixed-size (512c/64 overlap) + sentence-boundary |
| Embeddings | `embedder.py` — `all-MiniLM-L6-v2`, 384-dim, L2-normalised |
| Vector Store | `vector_store.py` — FAISS `IndexFlatIP` |
| Retrieval | `retriever.py` — hybrid: 0.70×dense + 0.30×sparse |
| Sparse Score | Manual IDF/TF keyword scorer (BM25-lite) |
| Prompt | `prompt_builder.py` — 3 iterations, context cap 3 000 chars |
| LLM | `llm_client.py` — Groq API via `requests` |
| Feedback | `feedback.py` — persistent chunk score boosts |

---

##  Quick Start

```bash
git clone https://github.com/Najart200/ai_10022300177.git
cd ai_10022300177
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your-free-key-from-console.groq.com"
streamlit run app.py
```

First run builds the FAISS index (~2–3 min). Subsequent runs load from disk (~5 sec).

---

##  Run Tests

```bash
python -m pytest tests/ -v
```

---

##  Deployment

Deployed on **Streamlit Community Cloud**: (https://najart200-ai-10022300177-app-6oddox.streamlit.app/)

---

##  Submission

**Email subject:** `CS4241 RAG Project Submission - Najart Rauf Awuni - 10022300177`
**Collaborator:** `GodwinDansoAcity` / `godwin.danso@acity.edu.gh`

---

##  Compliance

-  No LangChain / LlamaIndex / Haystack / any RAG framework
-  All core components manually implemented
-  Two datasets: Ghana Elections CSV + 2025 Budget PDF
-  Two chunking strategies with comparative analysis
-  Hybrid search extension (dense + manual sparse)
-  Three prompt iterations
-  Full pipeline logging (JSON per query)
-  Novel feature: feedback-driven retrieval improvement
-  Architecture diagram (Mermaid)
-  Adversarial testing templates
