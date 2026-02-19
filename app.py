"""
NyayaSathi AI — Streamlit Application
=======================================
Main entry point for the Legal Document Simplifier & Risk Analyzer.

Run with:
    streamlit run app.py

Architecture:
  • Sidebar  → mode selection (Groq / Ollama) + file upload
  • Tab 1    → Document Overview (classification + text preview)
  • Tab 2    → Clause Viewer with per-clause simplification
  • Tab 3    → Risk Analysis with heatmap and risk cards
  • Tab 4    → AI Chat (context-aware Q&A)
  • Tab 5    → Report (download-ready analysis)
"""

from __future__ import annotations

import os
import sys
import streamlit as st
import plotly.graph_objects as go

# ── Ensure project root is on sys.path ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from services.document_parser import parse_document
from services.classifier import classify_document
from services.clause_segmenter import segment_clauses
from services.risk_analyzer import RiskAnalyzer
from services.llm_router import LLMRouter
from services.rag_engine import RAGEngine
from embeddings.embedding_model import EmbeddingModel
from reports.report_generator import ReportGenerator
from utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=config.APP_TITLE,
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ══════════════════════════════════════════════════════════════════════
#  CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Header ── */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .main-header p {
        margin: 0.5rem 0 0 0;
        opacity: 0.85;
        font-size: 1.05rem;
    }

    /* ── Risk cards ── */
    .risk-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border-left: 5px solid;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(4px);
    }
    .risk-low    { border-color: #00c853; background: rgba(0,200,83,0.06); }
    .risk-medium { border-color: #ffc107; background: rgba(255,193,7,0.06); }
    .risk-high   { border-color: #ff5722; background: rgba(255,87,34,0.06); }
    .risk-critical { border-color: #d50000; background: rgba(213,0,0,0.08); }

    /* ── Metric badges ── */
    .metric-badge {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
    }
    .badge-low      { background: #00c853; color: white; }
    .badge-medium   { background: #ffc107; color: #333; }
    .badge-high     { background: #ff5722; color: white; }
    .badge-critical { background: #d50000; color: white; }

    /* ── Sidebar polish ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f23 0%, #1a1a3e 100%);
    }
    [data-testid="stSidebar"] * {
        color: #e0e0e0 !important;
    }

    /* ── Chat bubbles ── */
    .chat-user {
        background: linear-gradient(135deg, #0f3460, #16213e);
        color: white;
        padding: 1rem 1.2rem;
        border-radius: 16px 16px 4px 16px;
        margin: 0.5rem 0;
        max-width: 85%;
        margin-left: auto;
    }
    .chat-ai {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 1rem 1.2rem;
        border-radius: 16px 16px 16px 4px;
        margin: 0.5rem 0;
        max-width: 85%;
    }

    /* ── Clause expanders ── */
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* ── Tabs styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════

_DEFAULTS = {
    "doc_text": None,
    "doc_type": None,
    "clauses": None,
    "risk_results": None,
    "simplified_texts": {},
    "chat_history": [],
    "report_text": None,
    "rag_engine": None,
    "llm_router": None,
    "obligations": [],
    "deadlines": [],
    "negotiation_points": [],
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════
#  SERVICE INITIALISATION (cached)
# ══════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading LLM Router …")
def get_llm_router() -> LLMRouter:
    return LLMRouter()


@st.cache_resource(show_spinner="Loading Embedding Model …")
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel.get_instance()


@st.cache_resource(show_spinner="Building Legal Reference Index …")
def get_rag_engine() -> RAGEngine:
    model = get_embedding_model()
    engine = RAGEngine(embedding_model=model)

    # Try loading pre-built index
    if engine.load_index():
        return engine

    # Build from reference documents
    ref_dir = config.LEGAL_REFERENCES_DIR
    if os.path.isdir(ref_dir):
        docs = []
        for fname in os.listdir(ref_dir):
            fpath = os.path.join(ref_dir, fname)
            if os.path.isfile(fpath):
                with open(fpath, "r", encoding="utf-8") as f:
                    docs.append(f.read())
        if docs:
            engine.build_index(docs)
            engine.save_index()
    return engine


# ══════════════════════════════════════════════════════════════════════
#  DOCUMENT PROCESSING PIPELINE (defined before sidebar to avoid forward-ref)
# ══════════════════════════════════════════════════════════════════════

def _process_document(uploaded_file, llm_mode: str, router: LLMRouter):
    """Run the full analysis pipeline on the uploaded document."""
    progress = st.sidebar.progress(0, text="Starting analysis …")

    try:
        # Step 1 — Parse
        progress.progress(10, text="📄 Extracting text …")
        doc_text = parse_document(uploaded_file, filename=uploaded_file.name)
        st.session_state.doc_text = doc_text

        # Step 2 — Classify
        progress.progress(25, text="🏷️ Classifying document …")
        doc_type = classify_document(doc_text)
        st.session_state.doc_type = doc_type

        # Step 3 — Segment
        progress.progress(40, text="✂️ Segmenting clauses …")
        clauses = segment_clauses(doc_text)
        st.session_state.clauses = clauses

        # Step 4 — Risk Analysis
        progress.progress(55, text="⚠️ Analyzing risks …")
        analyzer = RiskAnalyzer()
        risk_results = analyzer.analyze_document(clauses, router, llm_mode)
        st.session_state.risk_results = risk_results

        # Step 5 — Extract obligations & deadlines via LLM
        progress.progress(75, text="📋 Extracting key details …")
        _extract_key_details(doc_text, router, llm_mode)

        # Step 6 — Add document to RAG index
        progress.progress(90, text="🧠 Indexing document …")
        rag = get_rag_engine()
        rag.add_document(doc_text, metadata={"source": uploaded_file.name})
        st.session_state.rag_engine = rag

        progress.progress(100, text="✅ Analysis complete!")
        logger.info("Document analysis complete: %s → %s", uploaded_file.name, doc_type)

    except Exception as exc:
        st.sidebar.error(f"❌ Analysis failed: {exc}")
        logger.error("Pipeline error: %s", exc, exc_info=True)


def _extract_key_details(doc_text: str, router: LLMRouter, mode: str):
    """Use LLM to extract obligations, deadlines, and negotiation points."""
    prompt = (
        "Analyze this legal document and extract:\n"
        "1. KEY OBLIGATIONS (list the main obligations for each party)\n"
        "2. IMPORTANT DEADLINES (list any dates, time limits, notice periods)\n"
        "3. NEGOTIATION POINTS (suggest 3-5 points that could be negotiated for better terms)\n\n"
        "Format your response as:\n"
        "OBLIGATIONS:\n- ...\n\n"
        "DEADLINES:\n- ...\n\n"
        "NEGOTIATION POINTS:\n- ...\n\n"
        f"Document text (first 3000 chars):\n{doc_text[:3000]}"
    )
    try:
        response = router.generate_response(prompt, mode=mode, max_tokens=1500)
        st.session_state.obligations = _parse_list_section(response, "OBLIGATIONS")
        st.session_state.deadlines = _parse_list_section(response, "DEADLINES")
        st.session_state.negotiation_points = _parse_list_section(response, "NEGOTIATION POINTS")
    except Exception as exc:
        logger.warning("Key detail extraction failed: %s", exc)


def _parse_list_section(text: str, section_name: str) -> list[str]:
    """Parse a bullet-list section from LLM output."""
    import re
    pattern = rf"{section_name}:\s*\n((?:[-•*]\s+.+\n?)*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return []
    items = []
    for line in match.group(1).strip().split("\n"):
        line = re.sub(r"^[-•*]\s+", "", line.strip())
        if line:
            items.append(line)
    return items


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚖️ NyayaSathi AI")
    st.caption(f"v{config.APP_VERSION}")
    st.divider()

    # Mode selection
    mode = st.radio(
        "🤖 **AI Mode**",
        options=["⚡ Fast (Groq)", "🔒 Private (Ollama)"],
        index=0,
        help="Fast mode uses Groq cloud for speed. Private mode keeps data local via Ollama.",
    )
    llm_mode = "fast" if "Fast" in mode else "private"

    # Availability check
    router = get_llm_router()
    if llm_mode == "fast":
        if router.is_groq_available():
            st.success("Groq API connected", icon="✅")
        else:
            st.warning("Set `GROQ_API_KEY` env variable to use Fast mode.", icon="⚠️")
    else:
        if router.is_ollama_available():
            st.success("Ollama server connected", icon="✅")
        else:
            st.warning("Start Ollama server to use Private mode.", icon="⚠️")

    st.divider()

    # File upload
    uploaded_file = st.file_uploader(
        "📄 **Upload Legal Document**",
        type=["pdf", "docx"],
        help="Supported formats: PDF, DOCX",
    )

    if uploaded_file and st.button("🔍 Analyze Document", use_container_width=True, type="primary"):
        _process_document(uploaded_file, llm_mode, router)

    st.divider()
    st.markdown(
        "💡 **Tip:** Upload a rental agreement, employment contract, "
        "business contract, or legal notice for AI-powered analysis."
    )


# ══════════════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════════════

st.markdown(
    f"""
    <div class="main-header">
        <h1>⚖️ {config.APP_TITLE}</h1>
        <p>{config.APP_SUBTITLE}</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════
#  MAIN CONTENT — TABS
# ══════════════════════════════════════════════════════════════════════

if st.session_state.doc_text is None:
    # Welcome state
    st.markdown("## 👋 Welcome!")
    st.markdown(
        "Upload a legal document using the sidebar to get started. "
        "NyayaSathi AI will help you understand complex legal language, "
        "identify risks, and generate a comprehensive analysis report."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 📄")
        st.markdown("**Rental Agreements**")
        st.caption("Understand tenant rights & obligations")
    with col2:
        st.markdown("### 💼")
        st.markdown("**Employment Contracts**")
        st.caption("Review salary, notice & non-compete terms")
    with col3:
        st.markdown("### 🏢")
        st.markdown("**Business Contracts**")
        st.caption("Analyze payment, liability & IP clauses")
    with col4:
        st.markdown("### 📜")
        st.markdown("**Legal Notices**")
        st.caption("Decode demands & response strategies")

    st.stop()


# ── Analysis tabs (shown after document is processed) ────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Overview",
    "📑 Clauses",
    "⚠️ Risk Analysis",
    "💬 AI Chat",
    "📊 Report",
])


# ──────────────────────────── TAB 1: Overview ────────────────────────

with tab1:
    st.markdown("## Document Overview")

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Document Type", st.session_state.doc_type or "—")
    with col_b:
        st.metric("Total Clauses", len(st.session_state.clauses or []))
    with col_c:
        risk = st.session_state.risk_results
        if risk:
            level = risk["risk_level"]
            badge_cls = f"badge-{level.lower()}"
            st.metric("Risk Score", f"{risk['overall_score']}/100")
            st.markdown(
                f'<span class="metric-badge {badge_cls}">{level}</span>',
                unsafe_allow_html=True,
            )

    st.divider()

    # Obligations & Deadlines
    if st.session_state.obligations or st.session_state.deadlines:
        col_ob, col_dl = st.columns(2)
        with col_ob:
            st.markdown("### 📌 Key Obligations")
            for ob in st.session_state.obligations:
                st.markdown(f"- {ob}")
        with col_dl:
            st.markdown("### ⏰ Deadlines")
            for dl in st.session_state.deadlines:
                st.markdown(f"- {dl}")
        st.divider()

    # Text preview
    with st.expander("📄 Full Document Text", expanded=False):
        st.text_area(
            "Extracted Text",
            value=st.session_state.doc_text[:5000] + ("…" if len(st.session_state.doc_text) > 5000 else ""),
            height=400,
            disabled=True,
        )


# ──────────────────────────── TAB 2: Clauses ─────────────────────────

with tab2:
    st.markdown("## Clause Viewer")
    clauses = st.session_state.clauses or []
    router = get_llm_router()

    if not clauses:
        st.info("No clauses to display.")
    else:
        for cl in clauses:
            title = cl.get("title") or f"Clause {cl['id']}"
            risk_score = ""
            if st.session_state.risk_results:
                for cr in st.session_state.risk_results.get("clause_risks", []):
                    if cr["clause_id"] == cl["id"]:
                        risk_score = f" — Risk: {cr['overall_score']}/100 ({cr['risk_level']})"
                        break

            with st.expander(f"**{title}**{risk_score}", expanded=False):
                st.markdown(cl["text"])
                st.divider()

                # Simplify button
                if st.button(f"🔍 Simplify", key=f"simplify_{cl['id']}"):
                    with st.spinner("Simplifying …"):
                        try:
                            rag = get_rag_engine()
                            context = rag.get_context(cl["text"][:500]) if rag.is_ready else ""
                            prompt = (
                                "Simplify the following legal clause into plain English that any "
                                "common citizen can understand. Be concise but cover all key points.\n\n"
                                f"Legal Clause:\n{cl['text']}\n\n"
                            )
                            if context and "No relevant context" not in context:
                                prompt += f"Relevant legal references:\n{context}\n\n"
                            prompt += "Plain English explanation:"

                            simplified = router.generate_response(
                                prompt=prompt,
                                mode=llm_mode,
                                system_prompt="You are a legal simplification expert. Convert complex legal language into simple, clear English.",
                            )
                            st.session_state.simplified_texts[cl["id"]] = simplified
                        except Exception as exc:
                            st.error(f"Simplification failed: {exc}")

                # Show cached simplification
                if cl["id"] in st.session_state.simplified_texts:
                    st.success("**Simplified Version:**")
                    st.markdown(st.session_state.simplified_texts[cl["id"]])


# ──────────────────────────── TAB 3: Risk Analysis ───────────────────

with tab3:
    st.markdown("## Risk Analysis")
    risk = st.session_state.risk_results

    if not risk:
        st.info("Run document analysis to see risk results.")
    else:
        # Overall score gauge
        col_gauge, col_summary = st.columns([1, 2])

        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk["overall_score"],
                title={"text": "Overall Risk Score", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#1a1a2e"},
                    "steps": [
                        {"range": [0, 25], "color": "#00c853"},
                        {"range": [25, 50], "color": "#ffc107"},
                        {"range": [50, 75], "color": "#ff5722"},
                        {"range": [75, 100], "color": "#d50000"},
                    ],
                    "threshold": {
                        "line": {"color": "white", "width": 3},
                        "thickness": 0.8,
                        "value": risk["overall_score"],
                    },
                },
            ))
            fig.update_layout(height=300, margin=dict(t=50, b=0, l=30, r=30))
            st.plotly_chart(fig, use_container_width=True)

        with col_summary:
            st.markdown(f"### Risk Level: **{risk['risk_level']}**")
            st.markdown(f"**Total clauses analyzed:** {len(risk.get('clause_risks', []))}")
            st.markdown(f"**High-risk clauses:** {len(risk.get('high_risk_clauses', []))}")

            if risk.get("high_risk_clauses"):
                st.markdown("#### ⚡ Immediate Attention Required:")
                for hr in risk["high_risk_clauses"]:
                    title = hr.get("clause_title") or f"Clause {hr['clause_id']}"
                    st.markdown(f"- **{title}** — Score: {hr['overall_score']}/100")

        st.divider()

        # Risk heatmap — bar chart of per-clause scores
        st.markdown("### 📊 Risk Heatmap")
        clause_risks = risk.get("clause_risks", [])
        if clause_risks:
            labels = [cr.get("clause_title") or f"Clause {cr['clause_id']}" for cr in clause_risks]
            scores = [cr["overall_score"] for cr in clause_risks]
            colors = [
                "#00c853" if s < 25 else "#ffc107" if s < 50 else "#ff5722" if s < 75 else "#d50000"
                for s in scores
            ]

            fig_bar = go.Figure(go.Bar(
                x=labels,
                y=scores,
                marker_color=colors,
                text=[f"{s}" for s in scores],
                textposition="outside",
            ))
            fig_bar.update_layout(
                yaxis_title="Risk Score",
                yaxis_range=[0, 110],
                height=400,
                margin=dict(t=20, b=80),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Detailed risk cards
        st.markdown("### 📋 Detailed Clause Risks")
        for cr in clause_risks:
            title = cr.get("clause_title") or f"Clause {cr['clause_id']}"
            level_class = f"risk-{cr['risk_level'].lower()}"
            st.markdown(
                f"""<div class="risk-card {level_class}">
                    <strong>{title}</strong> — Score: {cr['overall_score']}/100
                    <span class="metric-badge badge-{cr['risk_level'].lower()}">{cr['risk_level']}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            with st.expander(f"Details for {title}", expanded=False):
                for dim, data in cr.get("dimensions", {}).items():
                    label = dim.replace("_", " ").title()
                    st.markdown(f"- **{label}** ({data['score']}/100): {data['explanation']}")


# ──────────────────────────── TAB 4: AI Chat ─────────────────────────

with tab4:
    st.markdown("## 💬 AI Legal Assistant")
    st.caption(f"Mode: {'⚡ Fast (Groq)' if llm_mode == 'fast' else '🔒 Private (Ollama)'}")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">{msg["content"]}</div>', unsafe_allow_html=True)

    # Chat input
    question = st.chat_input("Ask anything about the document …")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.spinner("Thinking …"):
            try:
                rag = get_rag_engine()
                context = rag.get_context(question) if rag.is_ready else ""

                system = (
                    "You are NyayaSathi AI, a legal document assistant. "
                    "Answer the user's question based on the uploaded document and relevant legal references. "
                    "Be precise, cite specific clauses when possible, and explain in simple terms."
                )
                prompt_parts = [f"Question: {question}"]
                if st.session_state.doc_text:
                    preview = st.session_state.doc_text[:2000]
                    prompt_parts.append(f"\nDocument excerpt:\n{preview}")
                if context and "No relevant context" not in context:
                    prompt_parts.append(f"\nRelevant legal references:\n{context}")

                router = get_llm_router()
                answer = router.generate_response(
                    prompt="\n".join(prompt_parts),
                    mode=llm_mode,
                    system_prompt=system,
                )
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
            except Exception as exc:
                error_msg = f"Sorry, I encountered an error: {exc}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

        st.rerun()


# ──────────────────────────── TAB 5: Report ──────────────────────────

with tab5:
    st.markdown("## 📊 Analysis Report")

    if st.session_state.risk_results is None:
        st.info("Run document analysis to generate a report.")
    else:
        # Generate report on demand
        if st.button("📝 Generate Report", type="primary", use_container_width=True):
            with st.spinner("Generating report …"):
                gen = ReportGenerator()
                report = gen.generate_report(
                    doc_type=st.session_state.doc_type,
                    clauses=st.session_state.clauses,
                    risk_results=st.session_state.risk_results,
                    simplified_texts=st.session_state.simplified_texts,
                    obligations=st.session_state.obligations,
                    deadlines=st.session_state.deadlines,
                    negotiation_points=st.session_state.negotiation_points,
                )
                st.session_state.report_text = report

        # Display and download
        if st.session_state.report_text:
            st.markdown(st.session_state.report_text)
            st.divider()
            st.download_button(
                label="⬇️ Download Report (.md)",
                data=st.session_state.report_text,
                file_name="nyayasathi_analysis_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
