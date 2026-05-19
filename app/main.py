"""SecureAgentRAG — Streamlit application entry point."""

from __future__ import annotations

import streamlit as st

# ── Page Configuration (MUST be the first Streamlit call) ────────────────────
st.set_page_config(
    page_title="SecureAgentRAG",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ruff: noqa: E402
# ── Imports after set_page_config ────────────────────────────────────────────
import sys
from pathlib import Path

# Ensure the project root is on sys.path for imports
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from app.components.sidebar import PREDEFINED_USERS, render_sidebar
from app.views.audit import render_audit_page
from app.views.chat import render_chat_page
from app.views.evaluation import render_evaluation_page
from app.views.upload import render_upload_page
from utils.logging import setup_logging
from utils.observability import setup_tracing

# ── Initialize Logging ───────────────────────────────────────────────────────
setup_logging()

# ── Initialize Observability (Phoenix/OpenTelemetry) ─────────────────────────
_tracing_enabled = setup_tracing()


# ── Session State Initialization ─────────────────────────────────────────────
def _init_session_state() -> None:
    """Initialize all session state variables with safe defaults."""
    defaults = {
        "current_user": PREDEFINED_USERS[0],
        "chat_history": [],
        "audit_log": [],
        "inference_mode": "local",
        "selected_model": "qwen3:8b",
        "evaluation_data": [],
        "recent_uploads": [],
        "thread_id": None,
        "active_thread_id": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_session_state()

# ── Sidebar ──────────────────────────────────────────────────────────────────
render_sidebar()

# ── Main Content Area with Tabs ──────────────────────────────────────────────
tab_chat, tab_upload, tab_audit, tab_eval = st.tabs(
    [
        "💬 Chat",
        "📤 Upload",
        "📋 Audit Log",
        "📈 Evaluation",
    ]
)

with tab_chat:
    render_chat_page()

with tab_upload:
    render_upload_page()

with tab_audit:
    render_audit_page()

with tab_eval:
    render_evaluation_page()
