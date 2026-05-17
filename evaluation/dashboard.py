"""Evaluation data access layer for the Streamlit dashboard.

Provides helper functions to transform raw metric data into formats
suitable for Streamlit chart components (line_chart, bar_chart, metrics).
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from utils.logging import get_logger

if TYPE_CHECKING:
    from evaluation.custom_metrics import MetricsCollector

logger = get_logger(__name__)


def get_evaluation_data(metrics_collector: MetricsCollector) -> dict[str, Any]:
    """Retrieve structured evaluation data ready for Streamlit rendering.

    Combines summary statistics with time-series data formatted for
    direct use with Streamlit chart components.

    Args:
        metrics_collector: MetricsCollector instance with recorded data.

    Returns:
        Dictionary containing:
            - summary: Overall metrics summary
            - time_series: Timestamped data for line charts
            - distributions: Bucketed data for histograms
            - provider_comparison: Per-provider aggregated stats
    """
    summary = metrics_collector.get_summary()

    # Build time series data
    time_series = _build_time_series(metrics_collector)

    # Build distribution data
    distributions = _build_distributions(metrics_collector)

    return {
        "summary": summary,
        "time_series": time_series,
        "distributions": distributions,
        "provider_comparison": summary.get("providers", {}),
        "total_queries": summary.get("total_data_points", 0),
    }


def _build_time_series(collector: MetricsCollector) -> dict[str, list[dict[str, Any]]]:
    """Build time-series data from recorded metrics.

    Args:
        collector: MetricsCollector with data points.

    Returns:
        Dictionary with latency and confidence time-series lists.
    """
    latency_series: list[dict[str, Any]] = []
    confidence_series: list[dict[str, Any]] = []

    for point in collector.get_data_points():
        ts = datetime.fromtimestamp(point.timestamp).isoformat()
        if not point.rbac_blocked:
            latency_series.append({"timestamp": ts, "latency_ms": point.latency_ms})
            confidence_series.append({"timestamp": ts, "confidence": point.confidence})

    return {
        "latency": latency_series,
        "confidence": confidence_series,
    }


def _build_distributions(collector: MetricsCollector) -> dict[str, list[dict[str, Any]]]:
    """Build distribution/histogram data from recorded metrics.

    Args:
        collector: MetricsCollector with data points.

    Returns:
        Dictionary with latency and confidence distribution buckets.
    """
    # Latency distribution buckets (ms)
    latency_buckets = [
        (0, 500, "0-500ms"),
        (500, 1000, "500ms-1s"),
        (1000, 2000, "1-2s"),
        (2000, 5000, "2-5s"),
        (5000, float("inf"), "5s+"),
    ]

    latency_dist: list[dict[str, Any]] = []
    points = [p for p in collector.get_data_points() if not p.rbac_blocked]

    for low, high, label in latency_buckets:
        count = sum(1 for p in points if low <= p.latency_ms < high)
        latency_dist.append({"bucket": label, "count": count})

    # Confidence distribution
    confidence_buckets = [
        (0.0, 0.3, "Low (0-0.3)"),
        (0.3, 0.5, "Below Average (0.3-0.5)"),
        (0.5, 0.7, "Average (0.5-0.7)"),
        (0.7, 0.9, "Good (0.7-0.9)"),
        (0.9, 1.01, "Excellent (0.9-1.0)"),
    ]

    confidence_dist: list[dict[str, Any]] = []
    for low, high, label in confidence_buckets:
        count = sum(1 for p in points if low <= p.confidence < high)
        confidence_dist.append({"bucket": label, "count": count})

    return {
        "latency": latency_dist,
        "confidence": confidence_dist,
    }


def format_for_chart(
    data: list[dict[str, Any]],
    x_key: str,
    y_key: str,
) -> dict[str, list[Any]]:
    """Transform a list of dicts into x/y lists for Streamlit charts.

    Utility for converting structured data into the format expected
    by ``st.line_chart`` or ``st.bar_chart``.

    Args:
        data: List of dictionaries containing chart data.
        x_key: Key to use for x-axis values.
        y_key: Key to use for y-axis values.

    Returns:
        Dictionary with 'x' and 'y' lists extracted from the data.
    """
    return {
        "x": [item.get(x_key) for item in data],
        "y": [item.get(y_key) for item in data],
    }
