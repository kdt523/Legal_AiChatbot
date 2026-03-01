"""
streamlit_app.py — Legal RAG Assistant UI.

A polished Streamlit interface for the Legal RAG pipeline.
Features:
 • Chat-style Q&A with message history
 • Expandable source/chunk debug view
 • Pipeline status indicator
 • Session-state management
"""

import sys
import os
import time
from pathlib import Path

import streamlit as st

# ── Resolve project root so src/ imports work ─────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
SRC  = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

import config

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ─────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    color: #e6edf3;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #1c2130 100%);
    border-right: 1px solid #30363d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ── Header strip ─────────────────────────────────────────────── */
.legal-header {
    background: linear-gradient(90deg, #1a2340 0%, #1e3a5f 50%, #1a2340 100%);
    border: 1px solid #2d4a6e;
    border-radius: 12px;
    padding: 20px 28px;
    margin-bottom: 24px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.legal-header h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #58a6ff;
    margin: 0;
}
.legal-header p {
    color: #8b949e;
    margin: 6px 0 0 0;
    font-size: 0.95rem;
}

/* ── Chat messages ────────────────────────────────────────────── */
.msg-user {
    background: linear-gradient(135deg, #1e3a5f, #2a4a75);
    border: 1px solid #2d4a6e;
    border-radius: 12px 12px 2px 12px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #cae4ff;
    font-size: 0.97rem;
    line-height: 1.6;
}
.msg-assistant {
    background: linear-gradient(135deg, #161f2e, #1c2a3e);
    border: 1px solid #30363d;
    border-left: 4px solid #58a6ff;
    border-radius: 2px 12px 12px 12px;
    padding: 14px 18px;
    margin: 10px 0;
    color: #e6edf3;
    font-size: 0.97rem;
    line-height: 1.8;
}
.msg-meta {
    font-size: 0.78rem;
    color: #6e7681;
    margin-top: 8px;
}

/* ── Source badge ─────────────────────────────────────────────── */
.source-badge {
    display: inline-block;
    background: #1a3a5c;
    border: 1px solid #2d4a6e;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    color: #58a6ff;
    margin: 4px 4px 4px 0;
}

/* ── Status indicators ────────────────────────────────────────── */
.status-ready   { color: #3fb950; font-weight: 600; }
.status-loading { color: #f0883e; font-weight: 600; }
.status-error   { color: #f85149; font-weight: 600; }

/* ── Metrics ──────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ─────────────────────────────────────────────────────────────────────────────
if "pipeline"      not in st.session_state: st.session_state.pipeline      = None
if "pipeline_ok"   not in st.session_state: st.session_state.pipeline_ok   = False
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "total_queries" not in st.session_state: st.session_state.total_queries = 0
if "total_latency" not in st.session_state: st.session_state.total_latency = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline loader (cached across reruns)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_pipeline():
    """Load and return the LegalRAGPipeline (cached globally)."""
    from pipeline import LegalRAGPipeline  # type: ignore  # noqa: PLC0415
    p = LegalRAGPipeline()
    p._load_components()   # eagerly loads all models/indexes
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚖️ Legal RAG")
    st.markdown("---")

    # Pipeline status
    st.markdown("### Pipeline Status")
    status_placeholder = st.empty()

    if st.button("🔄 Load / Reload Pipeline", use_container_width=True):
        load_pipeline.clear()           # bust cache on manual reload
        st.session_state.pipeline_ok = False

    st.markdown("---")

    # Settings
    st.markdown("### Settings")
    show_chunks = st.checkbox("Show retrieved chunks (debug)", value=False)
    top_k_ui    = st.slider("Chunks to retrieve (top-K)", 1, 10,
                            config.TOP_K_RETRIEVAL, key="top_k_slider")

    st.markdown("---")

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history  = []
        st.session_state.total_queries = 0
        st.session_state.total_latency = 0.0
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#6e7681'>Built with BAAI/bge-large-en · "
        "BGE Reranker · Google Gemini · FAISS · BM25</small>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Initialise pipeline
# ─────────────────────────────────────────────────────────────────────────────

if not st.session_state.pipeline_ok:
    artifacts_exist = (
        (config.FAISS_INDEX_PATH / "index.bin").exists()
        and config.EMBEDDINGS_PATH.exists()
        and config.METADATA_PATH.exists()
    )

    if not artifacts_exist:
        status_placeholder.markdown(
            '<span class="status-error">⚠️ Artifacts missing</span>',
            unsafe_allow_html=True,
        )
        st.warning(
            "**Pipeline artifacts not found.**\n\n"
            "Run the ingestion pipeline first:\n"
            "```\n"
            "python src/ingestion.py\n"
            "python src/embeddings.py\n"
            "python src/vector_store.py\n"
            "```\n"
            "Then place your legal .txt / .json / .pdf files in `data/raw/`."
        )
        st.stop()

    status_placeholder.markdown(
        '<span class="status-loading">⏳ Loading pipeline…</span>',
        unsafe_allow_html=True,
    )
    try:
        pipeline = load_pipeline()
        st.session_state.pipeline    = pipeline
        st.session_state.pipeline_ok = True
        status_placeholder.markdown(
            '<span class="status-ready">✅ Ready</span>',
            unsafe_allow_html=True,
        )
    except Exception as exc:
        status_placeholder.markdown(
            '<span class="status-error">❌ Load failed</span>',
            unsafe_allow_html=True,
        )
        st.error(f"Failed to load pipeline: {exc}")
        st.stop()
else:
    status_placeholder.markdown(
        '<span class="status-ready">✅ Ready</span>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main content
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="legal-header">
  <h1>⚖️ Legal RAG Assistant</h1>
  <p>Ask questions about Indian law (IPC, Constitution, contracts, etc.) powered by
     hybrid retrieval + cross-encoder reranking + Google Gemini.</p>
</div>
""", unsafe_allow_html=True)

# ── Stats bar ─────────────────────────────────────────────────────────────────
c1, c2, c3 = st.columns(3)
c1.metric("📝 Questions Asked",  st.session_state.total_queries)
c2.metric("⚡ Avg Latency",
          f"{(st.session_state.total_latency / max(st.session_state.total_queries, 1)):.2f}s")
c3.metric("📚 Chunks Indexed",
          len(st.session_state.pipeline._bm25.chunks)
          if st.session_state.pipeline_ok and st.session_state.pipeline
          else "—")

st.markdown("---")

# ── Chat history ──────────────────────────────────────────────────────────────
for entry in st.session_state.chat_history:
    role    = entry["role"]
    content = entry["content"]
    sources = entry.get("sources", [])
    latency = entry.get("latency_s", None)
    chunks  = entry.get("chunks", [])

    if role == "user":
        st.markdown(
            f'<div class="msg-user"><strong>👤 You</strong><br>{content}</div>',
            unsafe_allow_html=True,
        )
    else:
        badges = "".join(
            f'<span class="source-badge">📄 {s}</span>' for s in sources
        )
        meta   = f'<div class="msg-meta">🕐 {latency}s  ·  Sources: {badges or "N/A"}</div>'
        st.markdown(
            f'<div class="msg-assistant"><strong>🤖 Assistant</strong><br>'
            f'{content}{meta}</div>',
            unsafe_allow_html=True,
        )

        if show_chunks and chunks:
            with st.expander(f"🔍 Retrieved chunks ({len(chunks)})", expanded=False):
                for i, chunk in enumerate(chunks, 1):
                    st.markdown(
                        f"**[{i}] {chunk.get('source','?')}** "
                        f"— chunk `{chunk.get('chunk_index', '?')}`"
                    )
                    st.text_area(
                        label="Chunk text",
                        value=chunk.get("text", ""),
                        height=120,
                        key=f"chunk_{id(entry)}_{i}",
                        disabled=True,
                        label_visibility="collapsed",
                    )

# ── Query input ───────────────────────────────────────────────────────────────
st.markdown("### Ask a Legal Question")
with st.form("query_form", clear_on_submit=True):
    user_input = st.text_area(
        label="Your question",
        placeholder="e.g. What is the punishment for murder under IPC Section 302?",
        height=100,
        label_visibility="collapsed",
    )
    submitted = st.form_submit_button("⚖️ Ask", use_container_width=True)

if submitted and user_input.strip():
    question = user_input.strip()

    # Add user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Retrieving context and generating answer…"):
        try:
            result  = st.session_state.pipeline.query(question)
            answer  = result["answer"]
            sources = result["sources"]
            latency = result["latency_s"]
            chunks  = result["chunks"]
        except Exception as exc:
            answer  = f"⚠️ Pipeline error: {exc}"
            sources = []
            latency = 0.0
            chunks  = []

    st.session_state.chat_history.append({
        "role":      "assistant",
        "content":   answer,
        "sources":   sources,
        "latency_s": latency,
        "chunks":    chunks,
    })
    st.session_state.total_queries += 1
    st.session_state.total_latency += latency
    st.rerun()

elif submitted:
    st.warning("Please type a question before submitting.")

# ─────────────────────────────────────────────────────────────────────────────
# Example prompts
# ─────────────────────────────────────────────────────────────────────────────
if not st.session_state.chat_history:
    st.markdown("---")
    st.markdown("#### 💡 Example questions to get started")
    examples = [
        "What is the punishment for murder under IPC?",
        "Explain the right to equality under the Indian Constitution.",
        "What constitutes criminal breach of trust?",
        "Define cognizable and non-cognizable offences.",
        "What are the essential elements of a valid contract?",
    ]
    cols = st.columns(len(examples))
    for col, ex in zip(cols, examples):
        col.markdown(
            f"<div style='background:#161b22;border:1px solid #30363d;"
            f"border-radius:8px;padding:10px;font-size:0.82rem;color:#8b949e;"
            f"cursor:default'>{ex}</div>",
            unsafe_allow_html=True,
        )
