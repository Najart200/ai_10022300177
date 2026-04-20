# Architecture Diagram & Justification — Part F

**Student:** Najart Rauf Awuni | **Index:** 10022300177

---

## Mermaid Architecture Diagram

```mermaid
flowchart TD
    subgraph DATA["📁 Knowledge Base (Offline Build)"]
        CSV["🗳️ Ghana Election CSV\ngithub.com/GodwinDansoAcity"]
        PDF["📄 2025 Budget PDF\nmofep.gov.gh"]
        CSV --> CLEAN
        PDF --> CLEAN
        CLEAN["🧹 data_loader.py\n• Strip non-ASCII\n• Normalise whitespace\n• Convert numeric columns"]
        CLEAN --> CHUNK
        CHUNK["✂️ chunker.py\n• Strategy A: Fixed-size 512c/64 overlap\n• Strategy B: Sentence-boundary ~600c max"]
        CHUNK --> EMBED
        EMBED["🔢 embedder.py\nSentenceTransformer\nall-MiniLM-L6-v2\n384-dim unit vectors"]
        EMBED --> FAISS["🗄️ vector_store.py\nFAISS IndexFlatIP\n(cosine via dot-product)\n+ metadata.json"]
    end

    subgraph QUERY["🔍 Query Pipeline (Online)"]
        USER["👤 User Query\nStreamlit Input"]
        USER --> QEMBED["🔢 embed_query()\nSame model — unit vector"]
        QEMBED --> HYBRID

        subgraph HYBRID["⚡ retriever.py — Hybrid Search"]
            DENSE["Dense Search\nFAISS top-4K\ncosine similarity"]
            SPARSE["Sparse Score\nManual TF-IDF\nKeywordScorer"]
            COMBINE["Combined Score\n0.70 × dense\n+ 0.30 × sparse\n+ feedback_boost"]
            DENSE --> COMBINE
            SPARSE --> COMBINE
        end

        COMBINE --> TOPK["Top-K Chunks\n(sorted by hybrid score)"]
        TOPK --> CTX["📝 prompt_builder.py\nContext window:\n≤3000 chars greedy\n+ source citations"]
        CTX --> PROMPT["Final Prompt\n(Template V3:\nstructured + hallucination guard)"]
        PROMPT --> LLM["🤖 llm_client.py\nGroq API\nllama-3.1-8b-instant\nvia raw HTTP requests"]
        LLM --> RESP["📤 Response"]
    end

    subgraph UI["🖥️ Streamlit UI — app.py"]
        RESP --> DISPLAY["Display:\n• Formatted answer\n• Retrieved chunks + scores\n• Exact prompt sent\n• Stage timings"]
        DISPLAY --> FB["👍👎 Feedback\nfeedback.py\nPersisted to data/feedback.json\nBoosts future retrieval"]
        FB --> COMBINE
    end

    subgraph LOG["📊 Logging (Part D)"]
        EMBED -.-> LOGGER["logs/query_*.json\nEvery stage:\ntimestamp + inputs + outputs + ms"]
        LLM -.-> LOGGER
    end

    style DATA    fill:#EDE9FE,stroke:#6D28D9,color:#1E1B4B
    style QUERY   fill:#F0FDF4,stroke:#16A34A,color:#1E1B4B
    style UI      fill:#FFF7ED,stroke:#EA580C,color:#1E1B4B
    style LOG     fill:#F0F9FF,stroke:#0284C7,color:#1E1B4B
    style HYBRID  fill:#FEFCE8,stroke:#CA8A04,color:#1E1B4B
```

---

## Why This Architecture Suits the Academic City Domain

### 1. Dual-source heterogeneous data
Academic City requires querying both **structured tabular data** (election results: parties, constituencies, vote counts) and **unstructured narrative text** (budget policy paragraphs). A single retrieval strategy fails both. The hybrid search architecture solves this by:
- Dense search capturing semantic intent ("fiscal responsibility" → budget deficit chunks)
- Sparse search capturing precise entity matches ("NDC, Ashanti, 2020" → exact CSV rows)

### 2. No hallucination by design
The `CONTEXT ONLY` instruction in Template V3 and the source-citation requirement mean the model is explicitly told it cannot invent facts. This is critical for a political/financial domain where wrong numbers have real consequences.

### 3. Feedback loop closes the quality gap
The `FeedbackStore` records which chunks students found helpful. This is especially valuable for domain-specific jargon (e.g. "E-Levy", "COCOBOD") where the generic embedding model may not initially rank well.

### 4. Stateless chunking + persistent index
Building the FAISS index once and loading it from disk on subsequent runs means the Streamlit app starts in ~2 seconds after the first build, making it practical for live demonstration.

### 5. Sentence-boundary chunking preserves budget policy context
The 2025 Budget PDF contains multi-sentence policy statements. Splitting at sentence boundaries ensures that "The government targets a primary balance surplus of 0.5% of GDP by 2027" is never split mid-sentence, preserving the complete numerical claim for retrieval.
