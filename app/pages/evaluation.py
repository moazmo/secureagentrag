"""Evaluation dashboard — metrics, performance tracking, and health checks."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.async_helpers import run_async
from utils.health import run_health_checks
from utils.logging import get_logger

logger = get_logger(__name__)


def render_evaluation_page() -> None:
    """Render the evaluation dashboard with metrics, charts, and tables."""
    st.title("📈 Evaluation Dashboard")
    st.caption(
        "Track RAG pipeline performance, confidence scores, latency metrics, and service health."
    )

    # ── Service Health ───────────────────────────────────────────────────────
    _render_health_checks()
    st.divider()

    eval_data = st.session_state.get("evaluation_data", [])

    if not eval_data:
        st.info(
            "No evaluation data yet. Ask questions in the Chat tab to generate performance metrics."
        )
        return

    df = pd.DataFrame(eval_data)

    # ── Ragas Quality Scores ─────────────────────────────────────────────────
    _render_ragas_scores()
    st.divider()

    # ── Overview Metrics ─────────────────────────────────────────────────────
    _render_overview_metrics(df)
    st.divider()

    # ── Charts ───────────────────────────────────────────────────────────────
    _render_charts(df)
    st.divider()

    # ── Model Comparison ─────────────────────────────────────────────────────
    _render_model_comparison(df)
    st.divider()

    # ── Recent Evaluations Table ─────────────────────────────────────────────
    _render_recent_evaluations(df)


def _render_health_checks() -> None:
    """Render service health status cards."""
    st.subheader("🏥 Service Health")

    with st.spinner("Checking services..."):
        report = run_async(run_health_checks())

    cols = st.columns(len(report.services))
    for col, svc in zip(cols, report.services, strict=False):
        with col:
            optional = getattr(svc, "optional", False)
            if svc.healthy:
                icon = "✅"
                value = "Healthy"
            elif optional:
                icon = "⚪"
                value = "Optional"
            else:
                icon = "❌"
                value = "Unhealthy"
            st.metric(
                label=f"{icon} {svc.name.upper()}",
                value=value,
                delta=f"{svc.latency_ms:.0f}ms" if svc.latency_ms > 0 else None,
                delta_color="off",
            )
            if svc.message:
                st.caption(svc.message[:120])

    if not report.overall_healthy:
        st.error(
            "⚠️ A required service is unreachable (Qdrant or Ollama). "
            "Check that Docker Compose and Ollama are running.",
            icon="⚠️",
        )


def _render_ragas_scores() -> None:
    """Render Ragas quality metrics if available."""
    ragas_scores = st.session_state.get("ragas_scores", [])

    if not ragas_scores:
        st.info(
            "No Ragas scores yet. Ragas evaluation runs automatically after each query when the package is installed."
        )
        return

    st.subheader("🎯 Ragas Quality Scores")

    import pandas as pd

    ragas_df = pd.DataFrame(ragas_scores)

    # Show latest scores
    latest = ragas_df.iloc[-1]
    cols = st.columns(4)
    metrics = [
        ("Faithfulness", latest.get("faithfulness")),
        ("Answer Relevancy", latest.get("answer_relevancy")),
        ("Context Precision", latest.get("context_precision")),
        ("Overall Score", latest.get("overall_score")),
    ]
    for col, (name, value) in zip(cols, metrics, strict=False):
        with col:
            if value is not None:
                st.metric(name, f"{value:.2f}")
            else:
                st.metric(name, "N/A")

    # Show trend chart if enough data
    if len(ragas_df) > 1:
        st.caption("Overall Score Trend")
        chart_df = ragas_df[["timestamp", "overall_score"]].copy()
        chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
        chart_df = chart_df.set_index("timestamp")
        st.line_chart(chart_df["overall_score"])


def _render_overview_metrics(df: pd.DataFrame) -> None:
    """Render overview metric cards.

    Args:
        df: Evaluation data DataFrame.
    """
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Queries", len(df))

    with col2:
        avg_confidence = df["confidence"].mean() if "confidence" in df.columns else 0.0
        st.metric("Avg Confidence", f"{avg_confidence:.0%}")

    with col3:
        avg_latency = df["latency_ms"].mean() if "latency_ms" in df.columns else 0.0
        st.metric("Avg Latency", f"{avg_latency:.0f}ms")

    with col4:
        blocked = len(df[~df["security_passed"]]) if "security_passed" in df.columns else 0
        st.metric("RBAC Blocks", blocked)


def _render_charts(df: pd.DataFrame) -> None:
    """Render performance charts.

    Args:
        df: Evaluation data DataFrame.
    """
    st.subheader("📊 Performance Charts")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Confidence Over Time",
            "Latency Distribution",
            "Queries by User",
            "Query Types",
        ]
    )

    with tab1:
        if "confidence" in df.columns and len(df) > 1:
            chart_df = df[["timestamp", "confidence"]].copy()
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"])
            chart_df = chart_df.set_index("timestamp")
            st.line_chart(chart_df["confidence"])
        else:
            st.info("Need at least 2 queries to show confidence trend.")

    with tab2:
        if "latency_ms" in df.columns and len(df) > 0:
            st.bar_chart(df["latency_ms"].values)
            st.caption("Each bar represents a query's processing time (ms)")
        else:
            st.info("No latency data available.")

    with tab3:
        if "user" in df.columns:
            user_counts = df["user"].value_counts()
            st.bar_chart(user_counts)
        else:
            st.info("No user data available.")

    with tab4:
        if "query_type" in df.columns:
            type_counts = df["query_type"].value_counts()
            st.bar_chart(type_counts)
        else:
            st.info("No query type data available.")


def _render_model_comparison(df: pd.DataFrame) -> None:
    """Render model comparison table if multiple models/modes were used.

    Args:
        df: Evaluation data DataFrame.
    """
    st.subheader("🔄 Model Comparison")

    if "model" not in df.columns or df["model"].nunique() < 2:
        st.info("Use different models (local and cloud) to see comparison data.")
        return

    comparison = (
        df.groupby("model")
        .agg(
            queries=("confidence", "count"),
            avg_confidence=("confidence", "mean"),
            avg_latency_ms=("latency_ms", "mean"),
            min_latency_ms=("latency_ms", "min"),
            max_latency_ms=("latency_ms", "max"),
        )
        .round(2)
    )

    comparison["avg_confidence"] = comparison["avg_confidence"].apply(lambda x: f"{x:.0%}")
    comparison["avg_latency_ms"] = comparison["avg_latency_ms"].apply(lambda x: f"{x:.0f}")
    comparison["min_latency_ms"] = comparison["min_latency_ms"].apply(lambda x: f"{x:.0f}")
    comparison["max_latency_ms"] = comparison["max_latency_ms"].apply(lambda x: f"{x:.0f}")

    st.dataframe(comparison, width="stretch")


def _render_recent_evaluations(df: pd.DataFrame) -> None:
    """Render a table of recent query evaluations.

    Args:
        df: Evaluation data DataFrame.
    """
    st.subheader("🕐 Recent Evaluations")

    display_cols = ["timestamp", "query", "confidence", "latency_ms", "query_type", "user", "model"]
    available_cols = [c for c in display_cols if c in df.columns]

    recent = df[available_cols].sort_values("timestamp", ascending=False).head(20)
    st.dataframe(recent, width="stretch", hide_index=True)
