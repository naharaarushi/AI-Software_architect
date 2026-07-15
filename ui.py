"""
AI Software Architecture Agent — Streamlit Dashboard
======================================================
Premium, dark, SaaS-style UI wrapped around an existing LangChain +
Mistral + ChromaDB RAG backend.

IMPORTANT: The backend logic below (LLM initialization, embedding model,
ChromaDB configuration, retriever configuration, prompt templates,
classifier logic, and the RAG pipeline itself) is the EXACT logic that
was provided. It has only been moved inside functions so it can be
called from Streamlit's reactive execution model — nothing about model
names, retriever search_type/search_kwargs, prompt text, or the
classification/retrieval/generation flow has been changed.

Run with:  streamlit run app.py
(Run it from the same folder that contains `system.txt`, the
`chroma-db/` persisted vector store, and your `.env` file.)
"""

import re
from datetime import datetime

import streamlit as st

# ──────────────────────────────────────────────────────────────────────────
# BACKEND IMPORTS — UNCHANGED
# ──────────────────────────────────────────────────────────────────────────
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv


# ════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AI Software Architect",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════
# STATIC UI CONFIG — section ordering / icons / tech categories / examples
# (UI-only constants — do not affect backend behavior)
# ════════════════════════════════════════════════════════════════════════
SECTION_ORDER = [
    "Project Overview",
    "Functional Requirements",
    "Recommended Tech Stack",
    "High-Level Architecture",
    "Core Services",
    "Database Design",
    "API Design",
    "Scalability Strategy",
    "Security Considerations",
    "Reliability Strategy",
    "Architecture Decisions",
    "Assumptions",
]

SECTION_ICONS = {
    "Project Overview": "🧭",
    "Functional Requirements": "✅",
    "Recommended Tech Stack": "🧱",
    "High-Level Architecture": "🏗️",
    "Core Services": "⚙️",
    "Database Design": "🗄️",
    "API Design": "🔌",
    "Scalability Strategy": "📈",
    "Security Considerations": "🔒",
    "Reliability Strategy": "🛡️",
    "Architecture Decisions": "🧩",
    "Assumptions": "💭",
}

TECH_CATEGORIES = ["Frontend", "Backend", "Database", "Cache", "Messaging"]

EXAMPLE_QUERIES = [
    "Design Netflix for 100M users",
    "Build a food delivery platform",
    "Create a payment gateway",
    "Design a ride-sharing app",
]


# ════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — dark, premium, SaaS-style theme
# ════════════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        :root {
            --bg-0:#08080c;
            --bg-1:#0d0d14;
            --bg-2:#13131c;
            --surface:#15151f;
            --border:rgba(255,255,255,0.08);
            --border-strong:rgba(255,255,255,0.14);
            --text-primary:#edeef2;
            --text-secondary:#9a9bab;
            --text-tertiary:#6b6c7e;
            --accent:#7c6cf6;
            --accent-2:#22d3ee;
            --accent-grad: linear-gradient(135deg,#7c6cf6 0%,#9b7cf9 50%,#22d3ee 100%);
            --success:#34d399;
            --danger:#f87171;
        }

        .stApp {
            background: radial-gradient(circle at 15% 0%, #14132a 0%, var(--bg-0) 45%) fixed;
            color: var(--text-primary);
        }

        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stToolbar"] { right: 1rem; }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, var(--bg-1) 0%, var(--bg-0) 100%);
            border-right: 1px solid var(--border);
        }
        [data-testid="stSidebar"] * { color: var(--text-primary); }

        h1, h2, h3, h4, h5 {
            color: var(--text-primary) !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }

        .hero-title {
            font-size: 2.6rem;
            font-weight: 800;
            background: var(--accent-grad);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.2rem;
            letter-spacing: -0.03em;
        }
        .hero-subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
            font-weight: 500;
            margin-bottom: 1.8rem;
        }

        .status-row {
            display:flex; align-items:center; gap:0.6rem;
            padding: 0.55rem 0.75rem;
            border-radius: 10px;
            background: var(--surface);
            border: 1px solid var(--border);
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }
        .status-dot {
            width:8px; height:8px; border-radius:50%;
            box-shadow: 0 0 8px currentColor;
            flex-shrink:0;
        }
        .status-dot.online { background: var(--success); color: var(--success); }
        .status-dot.offline { background: var(--danger); color: var(--danger); }
        .status-label { color: var(--text-primary); font-weight:600; }

        .meta-row {
            display:flex; justify-content:space-between; align-items:center;
            padding: 0.5rem 0.1rem;
            border-bottom: 1px solid var(--border);
            font-size: 0.82rem;
        }
        .meta-row:last-child { border-bottom: none; }
        .meta-key { color: var(--text-tertiary); }
        .meta-val {
            color: var(--text-primary); font-weight:600;
            font-family:'JetBrains Mono', monospace; font-size:0.78rem;
            text-align:right;
        }
        .meta-box {
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 0.3rem 0.8rem;
        }

        .sidebar-section-title {
            font-size: 0.72rem;
            font-weight: 700;
            color: var(--text-tertiary);
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin: 1.2rem 0 0.6rem 0;
        }

        .stButton button {
            border-radius: 10px !important;
            border: 1px solid var(--border-strong) !important;
            background: var(--surface) !important;
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            transition: all 0.15s ease;
        }
        .stButton button:hover {
            border-color: var(--accent) !important;
            color: var(--accent-2) !important;
            transform: translateY(-1px);
        }
        .stButton button[kind="primary"] {
            background: var(--accent-grad) !important;
            border: none !important;
            color: #08080c !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 24px rgba(124,108,246,0.35);
        }
        .stButton button[kind="primary"]:hover {
            filter: brightness(1.08);
            transform: translateY(-1px);
        }
        .stButton button:disabled {
            opacity: 0.4 !important;
            transform: none !important;
        }

        .stTextArea textarea {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px !important;
            color: var(--text-primary) !important;
            font-size: 0.95rem !important;
        }
        .stTextArea textarea:focus {
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 3px rgba(124,108,246,0.18) !important;
        }

        /* Architecture section cards (st.container(border=True)) */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(160deg, var(--surface) 0%, var(--bg-1) 100%) !important;
            border: 1px solid var(--border) !important;
            border-radius: 16px !important;
            padding: 0.3rem 0.5rem !important;
            margin-bottom: 1.1rem !important;
        }

        .card-header {
            display:flex; align-items:center; gap:0.55rem;
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text-primary);
            padding: 0.7rem 0.4rem 0.4rem 0.4rem;
            border-bottom: 1px solid var(--border);
            margin-bottom: 0.6rem;
        }
        .card-icon { font-size:1.2rem; }

        .tech-badge-row { display:flex; flex-wrap:wrap; gap:0.6rem; margin: 0.5rem 0.4rem 1rem 0.4rem; }
        .tech-badge {
            display:flex; flex-direction:column; gap:0.15rem;
            background: var(--bg-2);
            border: 1px solid var(--border-strong);
            border-radius: 12px;
            padding: 0.55rem 0.95rem;
            min-width: 115px;
        }
        .tech-badge-label {
            font-size: 0.68rem; text-transform:uppercase; letter-spacing:0.06em;
            color: var(--accent-2); font-weight:700;
        }
        .tech-badge-value { font-size: 0.92rem; font-weight:600; color: var(--text-primary); }

        .general-card {
            background: var(--surface);
            border: 1px solid var(--border);
            border-left: 3px solid var(--accent);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            color: var(--text-primary);
            font-size: 0.98rem;
        }

        .source-count {
            display:inline-block;
            background: rgba(124,108,246,0.15);
            color: var(--accent-2);
            border: 1px solid rgba(124,108,246,0.3);
            border-radius: 20px;
            padding: 0.3rem 0.9rem;
            font-size: 0.8rem;
            font-weight: 700;
            margin-bottom: 0.9rem;
        }

        details, .streamlit-expanderHeader {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
        }

        .divider-fade {
            height:1px;
            background: linear-gradient(90deg, transparent, var(--border-strong), transparent);
            margin: 1.8rem 0;
        }

        ::-webkit-scrollbar { width: 8px; height:8px; }
        ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 10px; }
        ::-webkit-scrollbar-track { background: transparent; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════
# BACKEND — UNCHANGED LOGIC (wrapped for Streamlit's caching model)
#
# Everything inside initialize_backend() is the exact backend that was
# provided: same LLM, same embedding model, same ChromaDB config, same
# retriever config (search_type / search_kwargs), same prompt templates,
# and same classifier prompt/chain. Nothing here has been altered.
# ════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def initialize_backend():
    load_dotenv()

    embed = MistralAIEmbeddings(model='mistral-embed')
    llm = ChatMistralAI(model='mistral-small-latest')

    vec_store = Chroma(
        persist_directory='chroma-db',
        embedding_function=embed
    )

    retriever = vec_store.as_retriever(
        search_type='mmr',
        search_kwargs={
            'k': 8,
            'lambda_mul': 0.3,
            'fetch_k': 20
        }
    )

    with open('system.txt') as f:
        system = f.read()

    prompt = ChatPromptTemplate.from_messages([
        ('system', system),
        ('human', '''Context : {context} 
     user requirments
{question}''')
    ])

    classifier_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
You are a query classifier.
Return ONLY ONE WORD.
ARCHITECTURE:
- System design requests
- App ideas
- Software development
- APIs
- Databases
- Cloud
- Scalability
- Technical architecture
GENERAL:
- Greetings
- Jokes
- Personal questions
- General knowledge
- Small talk
Examples:
Hi -> GENERAL
Hello -> GENERAL
How are you -> GENERAL
Tell me a joke -> GENERAL
Design Netflix -> ARCHITECTURE
Build a food delivery app -> ARCHITECTURE
Create a social media platform -> ARCHITECTURE
Design a payment gateway -> ARCHITECTURE
"""
        ),
        ("human", "{query}")
    ])

    classifier_chain = classifier_prompt | llm

    return {
        "embed": embed,
        "llm": llm,
        "vec_store": vec_store,
        "retriever": retriever,
        "prompt": prompt,
        "classifier_prompt": classifier_prompt,
        "classifier_chain": classifier_chain,
        "system": system,
    }


def run_pipeline(backend, query, status=None):
    """
    Same flow as the original while-loop body:
    classify -> retrieve -> branch on classification ->
    build context -> invoke prompt -> invoke llm.
    `status` is optional and only used to surface progress in the UI.
    """

    def log(msg):
        if status is not None:
            status.write(msg)

    log("🔎 Classifying query intent…")
    classification = backend["classifier_chain"].invoke({"query": query}).content.strip()
    log(f"Intent detected: **{classification.upper()}**")

    log("📚 Retrieving relevant knowledge chunks…")
    chunks_retrieved = backend["retriever"].invoke(query)
    log(f"Retrieved **{len(chunks_retrieved)}** chunks")

    if "ARCHITECTURE" not in classification.upper():
        return {
            "query": query,
            "classification": classification,
            "chunks": chunks_retrieved,
            "response": None,
            "general_message": "I can only help with software architecture and system design.",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    context = "\n\n".join(
        [
            f"Source: {doc.metadata['source']}\n{doc.page_content}"
            for doc in chunks_retrieved
        ]
    )

    log("🧠 Synthesizing architecture blueprint…")
    final_prompt = backend["prompt"].invoke({
        "context": context,
        "question": query
    })
    response = backend["llm"].invoke(final_prompt)
    log("✅ Architecture ready")

    return {
        "query": query,
        "classification": classification,
        "chunks": chunks_retrieved,
        "response": response.content,
        "general_message": None,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# ════════════════════════════════════════════════════════════════════════
# UI-ONLY HELPERS (parsing / formatting — do not touch backend logic)
# ════════════════════════════════════════════════════════════════════════
def safe_attr(obj, name, default="—"):
    try:
        val = getattr(obj, name, None)
        return val if val else default
    except Exception:
        return default


def parse_sections(text):
    """Split the LLM response into the 12 known architecture sections."""
    if not text:
        return {}
    escaped = [re.escape(s) for s in SECTION_ORDER]
    pattern = re.compile(
        r"(?im)^[#>\s\*\d\.\)\-]{0,6}(" + "|".join(escaped) + r")[\s\*:]*$"
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return {}
    sections = {}
    for i, m in enumerate(matches):
        raw_title = m.group(1)
        canonical = next((s for s in SECTION_ORDER if s.lower() == raw_title.lower()), raw_title)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        if content:
            sections[canonical] = content
    return sections


def extract_tech_stack(text):
    """Pull Frontend/Backend/Database/Cache/Messaging values out of the
    Recommended Tech Stack section for badge rendering."""
    badges = {}
    if not text:
        return badges
    for cat in TECH_CATEGORIES:
        m = re.search(
            rf"(?im)^[\s\*\-•\d\.\)]{{0,6}}{re.escape(cat)}\s*[:\-]\s*(.+)$",
            text,
        )
        if m:
            value = m.group(1).strip()
            value = re.sub(r"^\*+|\*+$", "", value).strip()
            value = value.split("(")[0].strip().rstrip(".")
            if value:
                badges[cat] = value
    return badges


# ════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════
def render_sidebar(backend, backend_error):
    with st.sidebar:
        st.markdown("### 🏛️ Architect ")
        st.caption("RAG-powered system design copilot")

        st.markdown('<div class="sidebar-section-title">System Status</div>', unsafe_allow_html=True)

        ok = backend is not None

        def status_html(label, healthy):
            dot = "online" if healthy else "offline"
            text = "Connected" if healthy else "Offline"
            return (
                f'<div class="status-row"><span class="status-dot {dot}"></span>'
                f'<span class="status-label">{label}</span>'
                f'<span style="margin-left:auto;color:var(--text-tertiary);font-size:0.75rem;">{text}</span></div>'
            )

        st.markdown(status_html("LLM Connected", ok), unsafe_allow_html=True)
        st.markdown(status_html("ChromaDB Connected", ok), unsafe_allow_html=True)
        st.markdown(status_html("Retriever Ready", ok), unsafe_allow_html=True)

        if backend_error:
            with st.expander("⚠️ Initialization error", expanded=True):
                st.code(backend_error, language="text")

        st.markdown('<div class="sidebar-section-title">Model Information</div>', unsafe_allow_html=True)

        if backend is not None:
            embed_model = safe_attr(backend["embed"], "model", "mistral-embed")
            retriever_type = getattr(backend["retriever"], "search_type", "mmr")
            search_kwargs = getattr(backend["retriever"], "search_kwargs", {}) or {}
            top_k = search_kwargs.get("k", "—")
            fetch_k = search_kwargs.get("fetch_k", "—")
            llm_model = safe_attr(backend["llm"], "model", "mistral-small-latest")
        else:
            embed_model = retriever_type = top_k = fetch_k = llm_model = "—"

        meta_rows = [
            ("Embedding Model", embed_model),
            ("LLM Model", llm_model),
            ("Retriever Type", str(retriever_type).upper()),
            ("Top K Value", top_k),
            ("Fetch K", fetch_k),
        ]
        rows_html = "".join(
            f'<div class="meta-row"><span class="meta-key">{k}</span><span class="meta-val">{v}</span></div>'
            for k, v in meta_rows
        )
        st.markdown(f'<div class="meta-box">{rows_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-section-title">Query History</div>', unsafe_allow_html=True)
        if not st.session_state.history:
            st.caption("No queries yet — generate your first architecture.")
        else:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.active_result = None
                st.rerun()
            for idx, item in enumerate(st.session_state.history):
                label = item["query"][:42] + ("…" if len(item["query"]) > 42 else "")
                icon = "💬" if item.get("general_message") else "🏗️"
                if st.button(f"{icon} {label}", key=f"hist_{idx}", use_container_width=True):
                    st.session_state.active_result = item
                    st.rerun()

        st.markdown('<div class="divider-fade"></div>', unsafe_allow_html=True)
        st.caption("Powered by Mistral · ChromaDB · LangChain")


# ════════════════════════════════════════════════════════════════════════
# QUERY SECTION
# ════════════════════════════════════════════════════════════════════════
def render_query_section(backend):
    st.markdown('<div class="hero-title">AI Software Architect</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-subtitle">Production-Ready System Design powered by RAG</div>',
        unsafe_allow_html=True,
    )

    st.session_state.setdefault("query_input", "")

    st.markdown("##### Try an example")
    cols = st.columns(len(EXAMPLE_QUERIES))
    for col, ex in zip(cols, EXAMPLE_QUERIES):
        with col:
            if st.button(ex, key=f"chip_{ex}", use_container_width=True):
                st.session_state.query_input = ex
                st.rerun()

    st.text_area(
        "Describe the system you want to design",
        key="query_input",
        height=150,
        placeholder=(
            "e.g. Design Netflix for 100M users, Build a food delivery platform, "
            "Create a payment gateway, Design a ride-sharing app…"
        ),
        label_visibility="collapsed",
    )

    gen_col, hint_col = st.columns([1, 3])
    with gen_col:
        generate_clicked = st.button(
            "🚀 Generate Architecture",
            type="primary",
            use_container_width=True,
            disabled=backend is None,
        )
    with hint_col:
        if backend is None:
            st.warning("Backend not connected — see the sidebar for the initialization error.", icon="⚠️")

    if generate_clicked:
        query = st.session_state.query_input.strip()
        if not query:
            st.warning("Please enter a system design query.")
            return

        with st.status("Generating architecture…", expanded=True) as status:
            result = run_pipeline(backend, query, status=status)
            status.update(
                label="Architecture generated ✅" if result["response"] else "Classified as a general query",
                state="complete",
            )

        st.session_state.history.insert(0, result)
        st.session_state.active_result = result
        st.rerun()


# ════════════════════════════════════════════════════════════════════════
# RESULTS SECTION
# ════════════════════════════════════════════════════════════════════════
def render_results(result):
    if result is None:
        st.markdown('<div class="divider-fade"></div>', unsafe_allow_html=True)
        st.caption("Generate an architecture above to see results here.")
        return

    st.markdown('<div class="divider-fade"></div>', unsafe_allow_html=True)

    top_l, top_r = st.columns([3, 1])
    with top_l:
        st.markdown(f"#### Results for: _{result['query']}_")
    with top_r:
        st.caption(result.get("timestamp", ""))

    if result.get("general_message"):
        st.markdown(
            f'<div class="general-card">💬 {result["general_message"]}</div>',
            unsafe_allow_html=True,
        )
        return

    response_text = result["response"] or ""
    sections = parse_sections(response_text)
    chunks = result.get("chunks", [])

    col_main, col_sources = st.columns([2.4, 1], gap="large")

    with col_main:
        if not sections:
            with st.container(border=True):
                st.markdown(
                    '<div class="card-header"><span class="card-icon">📄</span>Architecture Blueprint</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(response_text)
        else:
            ordered_titles = [t for t in SECTION_ORDER if t in sections]
            extra_titles = [t for t in sections if t not in SECTION_ORDER]
            for title in ordered_titles + extra_titles:
                content = sections.get(title, "").strip()
                if not content:
                    continue
                icon = SECTION_ICONS.get(title, "📄")
                with st.container(border=True):
                    st.markdown(
                        f'<div class="card-header"><span class="card-icon">{icon}</span>{title}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(content)
                    if title == "Recommended Tech Stack":
                        badges = extract_tech_stack(content)
                        if badges:
                            badge_html = "".join(
                                f'<div class="tech-badge"><span class="tech-badge-label">{k}</span>'
                                f'<span class="tech-badge-value">{v}</span></div>'
                                for k, v in badges.items()
                            )
                            st.markdown(f'<div class="tech-badge-row">{badge_html}</div>', unsafe_allow_html=True)

        st.markdown("##### Export")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.download_button(
                "⬇️ Download Report (.md)",
                data=response_text,
                file_name=f"architecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with ec2:
            with st.expander("📋 Copy Response", expanded=False):
                st.code(response_text, language="markdown")
                st.caption("Use the copy icon in the top-right of the code block above.")

    with col_sources:
        st.markdown(
            f'<div class="source-count">📚 {len(chunks)} chunks retrieved</div>',
            unsafe_allow_html=True,
        )
        if not chunks:
            st.caption("No source chunks for this query.")
        for doc in chunks:
            source_name = doc.metadata.get("source", "Unknown source")
            with st.expander(f"📄 {source_name}", expanded=False):
                st.markdown(doc.page_content)


# ════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════
def main():
    inject_css()

    if "history" not in st.session_state:
        st.session_state.history = []
    if "active_result" not in st.session_state:
        st.session_state.active_result = None

    backend = None
    backend_error = None
    try:
        backend = initialize_backend()
    except Exception as e:
        backend_error = f"{type(e).__name__}: {e}"

    render_sidebar(backend, backend_error)
    render_query_section(backend)
    render_results(st.session_state.active_result)


if __name__ == "__main__":
    main()