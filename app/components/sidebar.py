"""Sidebar component with user simulation and settings."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ingestion.metadata import UserContext
from utils.logging import get_logger

logger = get_logger(__name__)

# ── Predefined Users ─────────────────────────────────────────────────────────

PREDEFINED_USERS: list[dict[str, Any]] = [
    {
        "user_id": "admin_01",
        "org_id": "acme_corp",
        "roles": ["admin", "analyst", "viewer"],
        "clearance_level": 3,
        "display_name": "Admin User",
    },
    {
        "user_id": "analyst_01",
        "org_id": "acme_corp",
        "roles": ["analyst", "viewer"],
        "clearance_level": 2,
        "display_name": "Senior Analyst",
    },
    {
        "user_id": "viewer_01",
        "org_id": "acme_corp",
        "roles": ["viewer"],
        "clearance_level": 1,
        "display_name": "Junior Viewer",
    },
    {
        "user_id": "external_01",
        "org_id": "partner_inc",
        "roles": ["viewer"],
        "clearance_level": 1,
        "display_name": "External Consultant",
    },
]

LOCAL_MODELS: list[str] = ["qwen3:8b", "qwen3:14b"]
CLOUD_MODELS: list[str] = ["gpt-4o-mini", "claude-sonnet", "llama-3.3-70b"]


def render_sidebar() -> None:
    """Render the full sidebar with user selector, inference mode, and system status."""
    with st.sidebar:
        st.title("🔒 SecureAgentRAG")
        st.divider()

        # ── User Selector ────────────────────────────────────────────────────
        st.subheader("👤 User Simulation")
        user_names = [u["display_name"] for u in PREDEFINED_USERS]
        current_display = st.session_state.current_user.get("display_name", user_names[0])

        selected_name = st.selectbox(
            "Active User",
            user_names,
            index=user_names.index(current_display) if current_display in user_names else 0,
            key="user_selector",
        )

        # Update session state when user changes
        selected_user = next(u for u in PREDEFINED_USERS if u["display_name"] == selected_name)
        if st.session_state.current_user.get("user_id") != selected_user["user_id"]:
            st.session_state.current_user = selected_user

        # User info display
        user = st.session_state.current_user
        clearance_badges = {1: "🟢 Low", 2: "🟡 Medium", 3: "🔴 High"}
        st.caption(f"**ID:** {user['user_id']}")
        st.caption(f"**Org:** {user['org_id']}")
        st.caption(f"**Roles:** {', '.join(user['roles'])}")
        st.caption(f"**Clearance:** {clearance_badges.get(user['clearance_level'], '❓ Unknown')}")

        st.divider()

        # ── Inference Mode ───────────────────────────────────────────────────
        st.subheader("🧠 Inference Settings")
        mode = st.radio(
            "Inference Mode",
            ["Local (Ollama)", "Cloud"],
            index=0 if st.session_state.inference_mode == "local" else 1,
            key="inference_mode_radio",
        )
        st.session_state.inference_mode = "local" if mode == "Local (Ollama)" else "cloud"

        # Model selector based on mode
        models = LOCAL_MODELS if st.session_state.inference_mode == "local" else CLOUD_MODELS
        current_model = st.session_state.selected_model
        model_index = models.index(current_model) if current_model in models else 0

        selected_model = st.selectbox(
            "Model",
            models,
            index=model_index,
            key="model_selector",
        )
        st.session_state.selected_model = selected_model

        st.divider()

        # ── System Status ────────────────────────────────────────────────────
        st.subheader("📡 System Status")
        _render_system_status()


def _render_system_status() -> None:
    """Check and display system connectivity status for Ollama and Qdrant."""
    col1, col2 = st.columns(2)

    # Ollama status
    with col1:
        ollama_ok = _check_ollama_health()
        if ollama_ok:
            st.success("Ollama", icon="✅")
        else:
            st.error("Ollama", icon="❌")

    # Qdrant status
    with col2:
        qdrant_ok = _check_qdrant_health()
        if qdrant_ok:
            st.success("Qdrant", icon="✅")
        else:
            st.error("Qdrant", icon="❌")


def _check_ollama_health() -> bool:
    """Check if Ollama server is reachable.

    Returns:
        True if healthy, False otherwise.
    """
    try:
        import httpx

        response = httpx.get(
            f"{_get_ollama_url()}/api/tags",
            timeout=3.0,
        )
        return response.status_code == 200
    except Exception:
        return False


def _check_qdrant_health() -> bool:
    """Check if Qdrant server is reachable.

    Returns:
        True if healthy, False otherwise.
    """
    try:
        import httpx

        from config.settings import settings

        response = httpx.get(
            f"{settings.qdrant_url}/collections",
            timeout=3.0,
        )
        return response.status_code == 200
    except Exception:
        return False


def _get_ollama_url() -> str:
    """Get the Ollama base URL from settings.

    Returns:
        Ollama URL string.
    """
    from config.settings import settings

    return settings.ollama_url


def get_current_user_context() -> UserContext:
    """Convert the current session user to a UserContext Pydantic model.

    Returns:
        UserContext instance built from session state.
    """
    user = st.session_state.current_user
    return UserContext(
        user_id=user["user_id"],
        org_id=user["org_id"],
        roles=user["roles"],
        clearance_level=user["clearance_level"],
    )
