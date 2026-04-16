# =============================================================================
# capstone_streamlit.py — Legal Document Assistant UI
# Agentic AI Capstone 2026 | Dr. Kanthi Kiran Sirra
#
# Run with: streamlit run capstone_streamlit.py
#
# Design decisions:
# - ALL expensive initialisations inside @st.cache_resource — runs ONCE per session.
#   Without this, the embedding model downloads and ChromaDB rebuilds on every
#   user message (30-60 second delay per message). (Q7 answer)
# - st.session_state stores messages list and thread_id — both reset on
#   'New Conversation' button click.
# - from agent import build_graph, ask — notebook is the whiteboard, .py is the product
# =============================================================================

import streamlit as st
import uuid
import sys
import os

# Add parent directory to path if running from different location
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import build_graph, ask

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Legal Document Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #f8f9fa;
        border-left: 4px solid #2196F3;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        margin: 0.3rem 0;
        font-size: 0.85rem;
    }
    .metric-pass {
        border-left-color: #4CAF50;
    }
    .metric-fail {
        border-left-color: #f44336;
    }
    .source-tag {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        margin: 2px;
    }
    .warning-box {
        background: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CACHED RESOURCE — RUNS ONCE PER SESSION
# @st.cache_resource prevents model reloading on every user interaction.
# This is the correct pattern for LLM + embedding model initialisation in Streamlit.
# =============================================================================

@st.cache_resource
def load_agent():
    """
    Initialise LLM, embedder, ChromaDB collection, and compiled graph.
    Called once per session — cached by Streamlit.
    Supports both Groq (GROQ_API_KEY) and Google Gemini (GOOGLE_API_KEY).
    Returns the compiled app for use in ask().
    """
    groq_key   = st.secrets.get("GROQ_API_KEY",   os.environ.get("GROQ_API_KEY",   ""))
    gemini_key = st.secrets.get("GOOGLE_API_KEY",  os.environ.get("GOOGLE_API_KEY", ""))
    app, embedder, collection, llm = build_graph(groq_api_key=groq_key, gemini_api_key=gemini_key)
    return app

# =============================================================================
# SESSION STATE INITIALISATION
# thread_id: unique session identifier for MemorySaver multi-turn memory
# messages: display history for the chat interface
# Both reset when user clicks 'New Conversation'
# =============================================================================

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_meta" not in st.session_state:
    st.session_state.last_meta = {}

# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.markdown("## ⚖️ Legal Document Assistant")
    st.markdown("*Agentic AI Capstone 2026*")
    st.divider()

    st.markdown("### 📚 Knowledge Base Topics")
    topics = [
        "🔹 Void vs Voidable Contracts (ICA 1872)",
        "🔹 Doctrine of Frustration — Section 56",
        "🔹 NDA Clauses & Enforceability",
        "🔹 Non-Compete & Restraint of Trade",
        "🔹 CPC — Order VII R.11, Res Judicata",
        "🔹 Limitation Act — Articles & Section 18",
        "🔹 Legal Notice — Section 80 CPC",
        "🔹 Bail — Sections 436/437/438/439 CrPC",
        "🔹 Power of Attorney (Suraj Lamp ruling)",
        "🔹 Confidentiality & Injunctive Relief",
        "🔹 Digital Evidence — Section 65B",
        "🔹 Arbitration — Section 8/34/37, Seat vs Venue",
        "🔹 Consumer Protection Act 2019",
        "🔹 IPC — Corporate & Financial Fraud",
        "🔹 Specific Relief Act 2018 Amendment",
    ]
    for topic in topics:
        st.markdown(f"<small>{topic}</small>", unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🛠️ Tools Available")
    st.markdown("""
    <small>
    ⏱️ <b>DateTime Calculator</b><br>
    Calculates legal deadlines, limitation period expiry, notice period end dates, 
    and Section 18 acknowledgement resets.
    </small>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### ⚠️ Disclaimer")
    st.markdown("""
    <small>
    This assistant provides information from its knowledge base only. 
    It does not provide legal advice. Always consult a qualified advocate 
    for legal decisions. The assistant will never fabricate section numbers 
    or case names not in its knowledge base.
    </small>
    """, unsafe_allow_html=True)

    st.divider()

    # New Conversation button — resets thread_id and messages
    if st.button("🔄 New Conversation", use_container_width=True, type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.last_meta = {}
        st.rerun()

    st.markdown(
        f"<small>Session ID: <code>{st.session_state.thread_id[:8]}...</code></small>",
        unsafe_allow_html=True
    )

# =============================================================================
# MAIN INTERFACE
# =============================================================================

st.markdown('<div class="main-header">⚖️ Legal Document Assistant</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">AI-powered assistant for paralegals and junior lawyers — '
    'India-specific legal knowledge base | Powered by LangGraph + ChromaDB + Groq</div>',
    unsafe_allow_html=True
)

# Load the agent (cached)
try:
    app = load_agent()
    agent_ready = True
except Exception as e:
    st.error(f"Failed to initialise agent: {str(e)}")
    st.info("Please ensure GROQ_API_KEY is set in Streamlit secrets or environment variables.")
    agent_ready = False

# Show last response metadata in an expander
if st.session_state.last_meta:
    meta = st.session_state.last_meta
    with st.expander("📊 Last Response Metadata", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Route", meta.get("route", "—").upper())
        with col2:
            faith = meta.get("faithfulness", 0.0)
            st.metric(
                "Faithfulness",
                f"{faith:.2f}",
                delta="PASS" if faith >= 0.7 else "LOW",
                delta_color="normal" if faith >= 0.7 else "inverse"
            )
        with col3:
            st.metric("Eval Retries", meta.get("eval_retries", 0))

        sources = meta.get("sources", [])
        if sources:
            st.markdown("**Sources retrieved:**")
            source_html = " ".join([f'<span class="source-tag">{s[:50]}</span>' for s in sources])
            st.markdown(source_html, unsafe_allow_html=True)

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "⚖️"):
        st.markdown(msg["content"])

# Chat input
if agent_ready:
    # Suggested questions for first-time users
    if not st.session_state.messages:
        st.markdown("#### 💡 Try asking:")
        suggestions = [
            "What is the difference between a void and voidable contract under ICA 1872?",
            "Is a post-employment non-compete clause enforceable in India?",
            "My limitation period of 3 years started on August 10, 2021. The defendant acknowledged the debt on January 5, 2023. When does my new limitation period end?",
            "Can WhatsApp messages be used as evidence in Indian courts? What certificate is required?",
            "What is the difference between Seat and Venue in arbitration?",
        ]
        cols = st.columns(1)
        for suggestion in suggestions[:3]:
            if st.button(f"💬 {suggestion[:80]}...", key=suggestion, use_container_width=True):
                st.session_state.pending_question = suggestion
                st.rerun()

    # Handle pending question from suggestion buttons
    pending = st.session_state.get("pending_question", "")

    user_input = st.chat_input(
        "Ask a legal question (e.g., 'What are grounds for setting aside an arbitration award?')",
        disabled=not agent_ready
    )

    # Process input — either from chat_input or pending suggestion
    question = user_input or pending
    if pending:
        st.session_state.pending_question = ""

    if question:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        # Generate response
        with st.chat_message("assistant", avatar="⚖️"):
            with st.spinner("Researching legal knowledge base..."):
                try:
                    result = ask(
                        question=question,
                        thread_id=st.session_state.thread_id,
                        app=app
                    )

                    answer = result["answer"]
                    st.markdown(answer)

                    # Store metadata for display
                    st.session_state.last_meta = {
                        "route": result["route"],
                        "faithfulness": result["faithfulness"],
                        "eval_retries": result["eval_retries"],
                        "sources": result["sources"],
                    }

                    # Show inline route badge
                    route = result["route"]
                    faith = result["faithfulness"]
                    route_emoji = {"retrieve": "📚", "tool": "🛠️", "memory_only": "💭"}.get(route, "❓")
                    st.markdown(
                        f"<small>{route_emoji} Route: <b>{route.upper()}</b> | "
                        f"Faithfulness: <b>{faith:.2f}</b></small>",
                        unsafe_allow_html=True
                    )

                    # Store assistant message
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

        st.rerun()

else:
    st.info("⚠️ Agent not ready. Check GROQ_API_KEY configuration.")

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown(
    "<small>Legal Document Assistant | Agentic AI Capstone 2026 | "
    "Dr. Kanthi Kiran Sirra | Knowledge base covers India-specific legal provisions only.</small>",
    unsafe_allow_html=True
)
