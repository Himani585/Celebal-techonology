"""
DriveWise – Metadata-Aware Automotive RAG Assistant
────────────────────────────────────────────────────
Run with:  streamlit run app.py
"""

import os, json
import streamlit as st
import pandas as pd
import plotly.express as px
from config import VECTOR_STORE_PATH, BROCHURE_DIR

# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DriveWise – AI Car Assistant",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Header gradient ── */
.dw-header {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    padding: 2rem 2.5rem;
    border-radius: 14px;
    color: white;
    text-align: center;
    margin-bottom: 1.5rem;
}
.dw-header h1  { font-size: 2.4rem; margin: 0; letter-spacing: 1px; }
.dw-header p   { opacity: .8; margin: .4rem 0 0; font-size: 1rem; }

/* ── Source card ── */
.src-card {
    background: #f4f6fb;
    border-left: 4px solid #302b63;
    padding: .9rem 1rem;
    border-radius: 6px;
    margin: .5rem 0;
    font-size: .88rem;
}
.src-tag {
    display: inline-block;
    background: #302b63;
    color: white;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: .76rem;
    margin-right: 6px;
}

/* ── Metric card ── */
.metric-pill {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border-radius: 10px;
    padding: .6rem 1rem;
    text-align: center;
    font-size: .85rem;
}

/* ── Tip box ── */
.tip-box {
    background: #fffbea;
    border: 1px solid #f0c040;
    border-radius: 8px;
    padding: .8rem 1rem;
    font-size: .88rem;
    margin-top: .5rem;
}
</style>
""", unsafe_allow_html=True)

# ── Cached resource loaders ───────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI models …")
def load_rag():
    from rag_pipeline import DriveWiseRAG
    return DriveWiseRAG()

@st.cache_resource(show_spinner="Loading evaluator …")
def load_evaluator():
    from evaluator import RAGEvaluator
    return RAGEvaluator()

@st.cache_resource
def load_logger():
    from logger import QueryLogger
    return QueryLogger()

# ── Helpers ───────────────────────────────────────────────────────────────────

def vectorstore_ready() -> bool:
    return os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss"))

def load_catalog() -> dict:
    path = os.path.join(VECTOR_STORE_PATH, "catalog.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def score_color(v: float) -> str:
    if v >= 0.75: return "🟢"
    if v >= 0.50: return "🟡"
    return "🔴"

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dw-header">
  <h1>🚗 DriveWise</h1>
  <p>Metadata-Aware Automotive RAG Assistant &nbsp;|&nbsp;
     Ask anything about your car brochure</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    page = st.radio("", ["💬 Chat", "📊 Analytics", "📥 Ingest Brochures"],
                    label_visibility="collapsed")
    st.divider()

    catalog        = load_catalog() if vectorstore_ready() else {}
    selected_brand = selected_model = None
    enable_eval    = False
    show_debug     = False

    if vectorstore_ready() and catalog:
        st.markdown("### 🚘 Select Vehicle")
        brands         = sorted(catalog.keys())
        selected_brand = st.selectbox("Brand", [b.title() for b in brands])
        if selected_brand:
            models         = sorted(catalog[selected_brand.lower()])
            selected_model = st.selectbox("Model", [m.title() for m in models])

        st.divider()
        st.markdown("### ⚙️ Options")
        enable_eval = st.toggle("Enable Evaluation Metrics",
                                help="Adds ~2 s per query (3 LLM calls)")
        show_debug  = st.toggle("Show Debug Info")
    else:
        st.info("No brochures indexed yet.\nGo to **📥 Ingest Brochures**.")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 – CHAT
# ════════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":

    if not vectorstore_ready():
        st.info("👆 Please ingest at least one brochure PDF first (see **📥 Ingest Brochures**).")
        st.stop()

    if not catalog:
        st.warning("Vectorstore exists but catalog is empty. Re-ingest your brochures.")
        st.stop()

    # ── Session state ────────────────────────────────────────────────────────
    if "chat_history"   not in st.session_state: st.session_state.chat_history   = []
    if "active_vehicle" not in st.session_state: st.session_state.active_vehicle = ""

    vehicle_key = f"{selected_brand}|{selected_model}"
    if st.session_state.active_vehicle != vehicle_key:
        st.session_state.chat_history   = []
        st.session_state.active_vehicle = vehicle_key

    # ── Vehicle banner ───────────────────────────────────────────────────────
    c1, c2 = st.columns([5, 1])
    with c1:
        st.markdown(f"#### 🚘 {selected_brand} {selected_model}")
        st.caption("Ask anything — mileage, safety, dimensions, features, variants …")
    with c2:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()

    # ── Suggested questions ──────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown("**Suggested questions:**")
        suggestions = [
            f"What is the mileage of the {selected_model}?",
            f"List all safety features of the {selected_model}.",
            f"What engine options are available?",
            f"What are the dimensions and boot space?",
            f"Which variants are available and what are the key differences?",
        ]
        cols = st.columns(len(suggestions))
        for col, sug in zip(cols, suggestions):
            if col.button(sug, use_container_width=True):
                st.session_state._pending_query = sug
                st.rerun()

    # ── Chat history ─────────────────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🚗"):
                st.write(msg["content"])

                # Sources
                if msg.get("sources"):
                    with st.expander(
                        f"📄 {len(msg['sources'])} brochure section(s) used as context"
                    ):
                        for src in msg["sources"]:
                            st.markdown(f"""
<div class="src-card">
  <span class="src-tag">Chunk {src['chunk_number']}</span>
  <span class="src-tag">{src['section']}</span>
  <span class="src-tag">Page {src['page_number']}</span>
  <span class="src-tag">Score {src['rerank_score']}</span>
  <br><br>
  <em>{src['snippet']}</em><br>
  <small style="color:#888">📁 {src['source_file']} · {src['doc_version']}</small>
</div>
""", unsafe_allow_html=True)

                # Evaluation scores
                if msg.get("eval_scores"):
                    ev = msg["eval_scores"]
                    st.divider()
                    e1, e2, e3, e4 = st.columns(4)
                    e1.metric("Faithfulness",
                              f"{ev['faithfulness']:.0%}",
                              help="Is the answer grounded in the context?")
                    e2.metric("Context Relevance",
                              f"{ev['context_relevance']:.0%}",
                              help="Are the retrieved chunks relevant?")
                    e3.metric("Completeness",
                              f"{ev['answer_completeness']:.0%}",
                              help="Does the answer address the full question?")
                    e4.metric("Overall",
                              f"{ev['overall']:.0%}")

                # Debug
                if show_debug and msg.get("debug"):
                    with st.expander("🔧 Debug"):
                        st.json(msg["debug"])

    # ── Input ────────────────────────────────────────────────────────────────
    # Handle suggested-question button clicks
    pending = getattr(st.session_state, "_pending_query", None)
    if pending:
        del st.session_state._pending_query
        user_input = pending
    else:
        user_input = st.chat_input(
            f"Ask about the {selected_brand} {selected_model} …"
        )

    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("🔍 Searching brochure …"):
            logger = load_logger()
            try:
                rag    = load_rag()
                result = rag.query(user_input, selected_brand, selected_model)

                eval_scores = None
                if enable_eval:
                    ev_engine   = load_evaluator()
                    eval_scores = ev_engine.evaluate(
                        user_input, result["answer"], result["sources"]
                    )

                logger.log(result, eval_scores)

                st.session_state.chat_history.append({
                    "role":        "assistant",
                    "content":     result["answer"],
                    "sources":     result["sources"],
                    "eval_scores": eval_scores,
                    "debug": {
                        "retrieved_count": result["retrieved_count"],
                        "reranked_count":  result["reranked_count"],
                        "response_time_s": result["response_time"],
                    },
                })

            except Exception as e:
                logger.log(
                    {"query": user_input, "brand": selected_brand,
                     "model": selected_model},
                    error=e,
                )
                st.session_state.chat_history.append({
                    "role":    "assistant",
                    "content": f"⚠️ An error occurred: {e}",
                })

        st.rerun()

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 – ANALYTICS
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📊 Analytics":
    st.header("📊 Analytics Dashboard")

    try:
        logger = load_logger()
        stats  = logger.get_stats()
        logs   = logger.get_logs(200)

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Total Queries",    stats["total_queries"])
        k2.metric("Failed",           stats["failed_queries"])
        k3.metric("Success Rate",     f"{stats['success_rate']:.0%}")
        k4.metric("Avg Response",     f"{stats['avg_response_time']} s")
        k5.metric("Avg Quality",
                  f"{stats['avg_overall_score']:.0%}"
                  if stats["avg_overall_score"] else "N/A")

        st.divider()

        if not logs:
            st.info("No queries yet – start chatting!")
            st.stop()

        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Response Time Over Time")
            fig = px.line(df, x="timestamp", y="response_time",
                          color_discrete_sequence=["#302b63"],
                          labels={"response_time": "Seconds"})
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader("Queries by Brand / Model")
            bm = df.groupby(["brand", "model"]).size().reset_index(name="count")
            fig = px.bar(bm, x="model", y="count", color="brand",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig, use_container_width=True)

        eval_df = df.dropna(subset=["overall_score"])
        if not eval_df.empty:
            c3, c4 = st.columns(2)

            with c3:
                st.subheader("Quality Score Distribution")
                fig = px.histogram(eval_df, x="overall_score", nbins=20,
                                   color_discrete_sequence=["#764ba2"])
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

            with c4:
                st.subheader("Faithfulness vs Context Relevance")
                fig = px.scatter(eval_df, x="context_relevance", y="faithfulness",
                                 color="overall_score", size="response_time",
                                 hover_data=["query", "model"],
                                 color_continuous_scale="Viridis")
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Query Log")
        disp = df[["timestamp", "brand", "model", "query",
                   "response_time", "retrieved_count",
                   "reranked_count", "overall_score", "failed"]].copy()
        disp["timestamp"] = disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(disp, use_container_width=True, height=350)

    except Exception as e:
        st.error(f"Analytics error: {e}")

# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 – INGEST
# ════════════════════════════════════════════════════════════════════════════════
elif page == "📥 Ingest Brochures":
    st.header("📥 Ingest Car Brochures")

    st.markdown("""
<div class="tip-box">
<b>How it works:</b>
Upload a PDF brochure below, give it a brand + model name, and click <b>Ingest</b>.
DriveWise will extract text, chunk it by section, create vector embeddings,
and add it to the searchable index — ready to chat in seconds.
</div>
""", unsafe_allow_html=True)

    st.divider()

    # ── Upload form ──────────────────────────────────────────────────────────
    st.subheader("Upload a new brochure")
    c1, c2, c3 = st.columns(3)
    brand_in   = c1.text_input("Brand",   placeholder="e.g.  Hyundai")
    model_in   = c2.text_input("Model",   placeholder="e.g.  Creta")
    version_in = c3.text_input("Version", value="v1")

    uploaded = st.file_uploader("Select brochure PDF", type=["pdf"])

    if uploaded and brand_in.strip() and model_in.strip():
        if st.button("🚀 Ingest Brochure", type="primary"):
            os.makedirs(BROCHURE_DIR, exist_ok=True)
            filename = f"{brand_in.strip()}_{model_in.strip()}_{version_in.strip()}.pdf"
            filepath = os.path.join(BROCHURE_DIR, filename)

            with open(filepath, "wb") as f:
                f.write(uploaded.read())

            with st.spinner(f"Ingesting {brand_in} {model_in} …"):
                try:
                    from ingest import ingest_all_brochures
                    ingest_all_brochures()
                    st.cache_resource.clear()
                    st.success(
                        f"✅  {brand_in} {model_in} ingested successfully! "
                        "Head to **💬 Chat** to start asking questions."
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")
    elif uploaded:
        st.warning("Please enter Brand and Model before ingesting.")

    st.divider()

    # ── Current index ────────────────────────────────────────────────────────
    st.subheader("Currently indexed vehicles")
    if vectorstore_ready() and catalog:
        for brand, models in catalog.items():
            cols = st.columns([1, 4])
            cols[0].markdown(f"**{brand.title()}**")
            cols[1].write(", ".join(m.title() for m in models))
    else:
        st.info("No vehicles indexed yet. Upload a brochure above to get started.")
