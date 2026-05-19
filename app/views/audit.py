"""Audit log page — view all system activities."""

from __future__ import annotations

from datetime import UTC, datetime
from io import StringIO

import pandas as pd
import streamlit as st

from utils.logging import get_logger

logger = get_logger(__name__)


def render_audit_page() -> None:
    """Render the audit log page with filters, table, and export options."""
    st.title("📋 Audit Log")
    st.caption("View all system activities including queries, uploads, and access control events.")

    audit_data = st.session_state.get("audit_log", [])

    if not audit_data:
        st.info("No audit entries yet. Interact with the system to generate audit data.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(audit_data)

    # ── Summary Metrics ──────────────────────────────────────────────────────
    _render_summary_metrics(df)
    st.divider()

    # ── Filters ──────────────────────────────────────────────────────────────
    filtered_df = _render_filters(df)

    # ── Audit Table ──────────────────────────────────────────────────────────
    st.subheader("📊 Audit Entries")
    display_cols = [
        "timestamp",
        "user",
        "action",
        "query",
        "details",
        "sensitivity",
        "status",
        "latency_ms",
    ]
    available_cols = [c for c in display_cols if c in filtered_df.columns]

    st.dataframe(
        filtered_df[available_cols].sort_values("timestamp", ascending=False),
        width="stretch",
        hide_index=True,
    )

    # ── Export ───────────────────────────────────────────────────────────────
    st.divider()
    _render_export(filtered_df)


def _render_summary_metrics(df: pd.DataFrame) -> None:
    """Render summary metric cards from audit data.

    Args:
        df: Full audit log DataFrame.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_queries = len(df[df["action"] == "query"]) if "action" in df.columns else 0
        st.metric("Total Queries", total_queries)

    with col2:
        total_uploads = len(df[df["action"] == "upload"]) if "action" in df.columns else 0
        st.metric("Total Uploads", total_uploads)

    with col3:
        blocked = len(df[df["action"] == "query_blocked"]) if "action" in df.columns else 0
        st.metric("Blocked Attempts", blocked)

    with col4:
        if "latency_ms" in df.columns and len(df) > 0:
            avg_latency = df["latency_ms"].mean()
            st.metric("Avg Latency", f"{avg_latency:.0f}ms")
        else:
            st.metric("Avg Latency", "N/A")


def _render_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render filter controls and return filtered DataFrame.

    Args:
        df: Full audit log DataFrame.

    Returns:
        Filtered DataFrame based on user selections.
    """
    st.subheader("🔍 Filters")
    col1, _col2, col3 = st.columns(3)

    filtered = df.copy()

    with col1:
        if "user" in df.columns:
            users = ["All", *sorted(df["user"].unique().tolist())]
            selected_user = st.selectbox("User", users, key="audit_user_filter")
            if selected_user != "All":
                filtered = filtered[filtered["user"] == selected_user]

    with _col2:
        if "action" in df.columns:
            actions = ["All", *sorted(df["action"].unique().tolist())]
            selected_action = st.selectbox("Action Type", actions, key="audit_action_filter")
            if selected_action != "All":
                filtered = filtered[filtered["action"] == selected_action]

    with col3:
        if "status" in df.columns:
            statuses = ["All", *sorted(df["status"].unique().tolist())]
            selected_status = st.selectbox("Status", statuses, key="audit_status_filter")
            if selected_status != "All":
                filtered = filtered[filtered["status"] == selected_status]

    st.caption(f"Showing {len(filtered)} of {len(df)} entries")
    return filtered


def _render_export(df: pd.DataFrame) -> None:
    """Render CSV export button.

    Args:
        df: DataFrame to export.
    """
    col1, _col2 = st.columns([1, 3])
    with col1:
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Export as CSV",
            data=csv_data,
            file_name=f"audit_log_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            width="stretch",
        )
