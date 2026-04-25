"""
app.py — ACity RAG Chatbot — Streamlit UI
Theme: Purple + White | Animated thinking panda mascot

Student: Najart Rauf Awuni | Index: 10022300177
Course: CS4241 Introduction to Artificial Intelligence — 2026
"""

import streamlit as st
from pathlib import Path
from urllib.parse import quote
from src.utils import setup_logging, truncate, score_colour
from src.prompt_builder import TEMPLATES, DEFAULT_TEMPLATE

setup_logging()

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ACity Scholar | RAG Chatbot",
    page_icon="🐼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  (purple + white palette, animated panda SVG)
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css():
    st.markdown("""
    <style>
    /* ── Root palette ── */
    :root {
        --purple-900: #4C1D95;
        --purple-700: #6D28D9;
        --purple-500: #8B5CF6;
        --purple-300: #C4B5FD;
        --purple-100: #EDE9FE;
        --white:      #FFFFFF;
        --off-white:  #FAFAFA;
        --text-dark:  #1E1B4B;
    }

    /* Hide Streamlit default chrome */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── App background ── */
    .stApp {
        background: linear-gradient(135deg, #F5F3FF 0%, #FFFFFF 60%, #EDE9FE 100%);
        font-family: 'Segoe UI', system-ui, sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #4C1D95 0%, #6D28D9 100%);
        color: #FFFFFF;
    }
    section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stSlider { color: var(--text-dark) !important; }

    /* Sidebar label overrides */
    section[data-testid="stSidebar"] label { color: #DDD6FE !important; }

    /* ── Header banner ── */
    .acity-header {
        background: linear-gradient(90deg, #4C1D95 0%, #7C3AED 50%, #A855F7 100%);
        color: #FFFFFF;
        padding: 18px 28px 14px;
        border-radius: 14px;
        margin-bottom: 18px;
        display: flex;
        align-items: center;
        gap: 18px;
        box-shadow: 0 4px 24px rgba(109,40,217,0.25);
    }
    .acity-header h1 {
        margin: 0; font-size: 1.65rem; font-weight: 700;
        text-shadow: 0 1px 6px rgba(0,0,0,0.2);
    }
    .acity-header p {
        margin: 4px 0 0; font-size: 0.82rem; opacity: 0.88;
    }

    /* ── Chat bubbles ── */
    .user-bubble {
        background: linear-gradient(135deg, #7C3AED, #6D28D9);
        color: #FFFFFF;
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0 8px 60px;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(109,40,217,0.3);
    }
    .bot-bubble {
        background: #FFFFFF;
        color: #1E1B4B;
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 60px 8px 0;
        font-size: 0.95rem;
        border: 1.5px solid #DDD6FE;
        box-shadow: 0 2px 10px rgba(109,40,217,0.08);
        line-height: 1.6;
    }

    /* ── Retrieved chunk card ── */
    .chunk-card {
        background: #FAFAFA;
        border-left: 4px solid #8B5CF6;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        color: #374151;
    }
    .chunk-score {
        font-weight: 700;
        font-size: 0.78rem;
        border-radius: 20px;
        padding: 2px 8px;
        display: inline-block;
    }
    .chunk-source {
        font-size: 0.75rem;
        color: #6D28D9;
        font-style: italic;
    }

    /* ── Prompt display ── */
    .prompt-box {
        background: #F3F0FF;
        border: 1.5px dashed #A78BFA;
        border-radius: 8px;
        padding: 12px;
        font-family: 'Courier New', monospace;
        font-size: 0.76rem;
        color: #4C1D95;
        white-space: pre-wrap;
        word-break: break-word;
        max-height: 280px;
        overflow-y: auto;
    }

    /* ── Feedback buttons ── */
    .stButton > button {
        border-radius: 20px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    /* ── Score pill ── */
    .score-pill {
        display: inline-block;
        border-radius: 999px;
        padding: 1px 10px;
        font-size: 0.75rem;
        font-weight: 700;
        color: #FFFFFF;
    }

    /* ── Status dot ── */
    .status-dot {
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        background: #22c55e;
        animation: pulse-dot 1.8s ease-in-out infinite;
    }
    @keyframes pulse-dot {
        0%,100% { opacity:1; transform:scale(1); }
        50%      { opacity:.5; transform:scale(1.3); }
    }

    /* ── Thinking panda float ── */
    .panda-wrap {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 4px 0 0;
    }

    /* ── Welcome card ── */
    .welcome-card {
        background: linear-gradient(135deg, #EDE9FE, #FFFFFF);
        border: 2px solid #DDD6FE;
        border-radius: 16px;
        padding: 30px 28px;
        text-align: center;
        margin: 20px auto;
        max-width: 560px;
    }
    .welcome-card h2 { color: #4C1D95; font-size: 1.35rem; margin-bottom: 8px; }
    .welcome-card p  { color: #6B7280; font-size: 0.9rem; }
    .welcome-card ul { text-align: left; color: #374151; font-size: 0.88rem; margin-top:12px; }

    /* ── Stage timing badge ── */
    .timing-badge {
        background: #EDE9FE;
        color: #6D28D9;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.72rem;
        font-weight: 600;
        display: inline-block;
        margin: 2px 4px;
    }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD PANDA SVG
# ─────────────────────────────────────────────────────────────────────────────
def _load_panda_svg() -> str:
    svg_path = Path(__file__).parent / "assets" / "panda.svg"
    if svg_path.exists():
        return svg_path.read_text(encoding="utf-8")
    return ""


def _panda_img_html(panda_svg: str, width: int = 250) -> str:
    """Render SVG safely via data URI to avoid markdown SVG sanitisation."""
    if not panda_svg:
        return ""
    encoded = quote(panda_svg)
    return (
        f'<div class="panda-wrap">'
        f'<img src="data:image/svg+xml;utf8,{encoded}" '
        f'alt="Thinking panda" width="{width}" />'
        f'</div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
#  PIPELINE SINGLETON  (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline(strategy: str = "sentence"):
    from src.pipeline import RAGPipeline
    return RAGPipeline(strategy=strategy)


# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
def _render_sidebar(panda_svg: str) -> dict:
    with st.sidebar:
        # Panda mascot
        if panda_svg:
            st.markdown(_panda_img_html(panda_svg, width=230), unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Settings")

        chunk_strategy = st.selectbox(
            "Chunking Strategy",
            ["sentence", "fixed"],
            help="Sentence: preserves meaning (PDF). Fixed: uniform size (CSV)."
        )

        template_key = st.selectbox(
            "Prompt Template",
            list(TEMPLATES.keys()),
            index=list(TEMPLATES.keys()).index(DEFAULT_TEMPLATE),
            help="V3 is the best-performing iteration (see docs)."
        )

        top_k = st.slider("Top-K Chunks", min_value=1, max_value=10, value=5)

        use_hybrid = st.toggle("Hybrid Search", value=True,
                               help="Combines dense (semantic) + sparse (keyword) scores.")

        temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.2, 0.05,
                                help="Lower = more factual; higher = more creative.")

        st.markdown("---")
        st.markdown("### 📊 Feedback Stats")
        try:
            pipeline = get_pipeline(chunk_strategy)
            stats    = pipeline.feedback.stats()
            col1, col2 = st.columns(2)
            col1.metric("👍 Likes",    stats["likes"])
            col2.metric("👎 Dislikes", stats["dislikes"])
            st.caption(f"Chunks boosted: {stats['chunks_boosted']} | Penalised: {stats['chunks_penalised']}")
        except Exception:
            st.caption("Stats unavailable until index is built.")

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("""
        **Student:** Najart Rauf Awuni
        **Index:** 10022300177
        **Course:** CS4241 AI — 2026
        **Datasets:**
        • Ghana Election Results (CSV)
        • 2025 Budget Statement (PDF)
        """)

        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()

        if st.button("🔄 Rebuild Index", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.chat = []
            st.rerun()

    return {
        "chunk_strategy": chunk_strategy,
        "template_key":   template_key,
        "top_k":          top_k,
        "use_hybrid":     use_hybrid,
        "temperature":    temperature,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  CHUNK CARDS (retrieved context display — Part D)
# ─────────────────────────────────────────────────────────────────────────────
def _render_chunk_cards(chunks: list[dict]):
    st.markdown("**📚 Retrieved Chunks (with scores)**")
    for i, chunk in enumerate(chunks, 1):
        score      = chunk.get("score", 0)
        dense_s    = chunk.get("dense_score", score)
        sparse_s   = chunk.get("sparse_score", 0)
        colour     = score_colour(score)
        source     = chunk.get("source", "unknown")
        page       = chunk.get("page", "")
        page_tag   = f" — Page {page}" if page else ""
        cid        = chunk.get("chunk_id", "?")

        score_html = (
            f'<span class="score-pill" style="background:{colour}">'
            f'{score:.4f}</span>'
        )
        st.markdown(
            f'<div class="chunk-card">'
            f'<b>#{i}</b> {score_html} '
            f'<span style="font-size:0.72rem;color:#6B7280;">'
            f'dense={dense_s:.3f} | sparse={sparse_s:.3f} | chunk_id={cid}'
            f'</span><br>'
            f'<span class="chunk-source">{source}{page_tag}</span><br><br>'
            f'{truncate(chunk["text"], 250)}'
            f'</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _inject_css()
    panda_svg = _load_panda_svg()

    # Session state
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "last_run" not in st.session_state:
        st.session_state.last_run = None

    # Sidebar
    cfg = _render_sidebar(panda_svg)

    # ── Header banner ──────────────────────────────────────────────────────
    st.markdown(
        '<div class="acity-header">'
        '<div style="font-size:2.5rem">🐼</div>'
        '<div>'
        '<h1>ACity Scholar — RAG Chatbot</h1>'
        '<p><span class="status-dot"></span> '
        'Knowledge base: Ghana Elections (CSV) + 2025 Budget Statement (PDF) '
        '| CS4241 AI Project — Najart Rauf Awuni (10022300177)</p>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )

    # ── Index build progress ───────────────────────────────────────────────
    with st.spinner("🔄 Loading knowledge base (first run may take 2–3 min)…"):
        try:
            pipeline = get_pipeline(cfg["chunk_strategy"])
            st.success(
                f"✅ Index ready — {pipeline.store.index.ntotal} chunks indexed.",
                icon="🗄️"
            )
        except Exception as e:
            st.error(f"❌ Failed to build index: {e}")
            st.info("Check your internet connection and try rebuilding the index.")
            return

    # ── Welcome state (empty chat) ─────────────────────────────────────────
    if not st.session_state.chat:
        st.markdown(
            '<div class="welcome-card">'
            f'{_panda_img_html(panda_svg, width=260)}'
            '<h2>Hello! I\'m ACity Scholar 🐼</h2>'
            '<p>I can answer questions about Ghana\'s elections and the 2025 Budget Statement '
            'using <em>only</em> the provided knowledge base — no hallucinations!</p>'
            '<ul>'
            '<li>🗳️ "Who won the most parliamentary seats in 2020?"</li>'
            '<li>💰 "What is the 2025 budget deficit target?"</li>'
            '<li>📊 "Which region had the highest voter turnout?"</li>'
            '<li>🏛️ "What are the key education allocations in the 2025 budget?"</li>'
            '</ul>'
            '</div>',
            unsafe_allow_html=True
        )

    # ── Chat history display ───────────────────────────────────────────────
    for turn in st.session_state.chat:
        st.markdown(
            f'<div class="user-bubble">🧑 {turn["question"]}</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div class="bot-bubble">🐼 {turn["response"]}</div>',
            unsafe_allow_html=True
        )

        # Feedback buttons (Part G)
        fb_key_like    = f"like_{turn['idx']}"
        fb_key_dislike = f"dislike_{turn['idx']}"
        col_fb1, col_fb2, col_fb_sp = st.columns([1, 1, 8])
        if col_fb1.button("👍", key=fb_key_like, help="Mark this response as helpful"):
            used_ids = [c.get("chunk_id") for c in turn.get("chunks_used", [])]
            pipeline.feedback.record(turn["question"], used_ids, liked=True)
            st.toast("Thanks! Retrieval will improve for similar queries.", icon="✅")
        if col_fb2.button("👎", key=fb_key_dislike, help="Mark this response as unhelpful"):
            used_ids = [c.get("chunk_id") for c in turn.get("chunks_used", [])]
            pipeline.feedback.record(turn["question"], used_ids, liked=False)
            st.toast("Got it! Those chunks will be ranked lower next time.", icon="📉")

        st.markdown("---")

    # ── Query input ────────────────────────────────────────────────────────
    with st.form("query_form", clear_on_submit=True):
        query = st.text_input(
            "Ask a question about Ghana's elections or the 2025 budget…",
            placeholder="e.g. What percentage of the 2025 budget goes to education?",
        )
        submitted = st.form_submit_button("🔍 Ask ACity Scholar", use_container_width=True)

    if submitted and query.strip():
        with st.spinner("🐼 Thinking…"):
            run = pipeline.query(
                question=query.strip(),
                top_k=cfg["top_k"],
                template_key=cfg["template_key"],
                use_hybrid=cfg["use_hybrid"],
                temperature=cfg["temperature"],
            )

        # Save to chat history
        turn_entry = {
            "idx":                  len(st.session_state.chat),
            "question":             run["question"],
            "response":             run["response"],
            "retrieved_chunks":     run.get("retrieved_chunks", []),
            "chunks_used":          run.get("chunks_used", []),
            "prompt":               run.get("prompt", ""),
            "stage1_embed_ms":      run.get("stage1_embed_ms", 0),
            "stage2_retrieval_ms":  run.get("stage2_retrieval_ms", 0),
            "stage3_prompt_ms":     run.get("stage3_prompt_ms", 0),
            "stage4_llm_ms":        run.get("stage4_llm_ms", 0),
        }
        st.session_state.chat.append(turn_entry)
        st.rerun()


if __name__ == "__main__":
    main()
