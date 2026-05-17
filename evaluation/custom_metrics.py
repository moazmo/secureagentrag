"""Custom evaluation metrics for SecureAgentRAG.

Provides in-memory metric collection with optional SQLite persistence,
statistical aggregation, and provider comparison capabilities for
monitoring RAG pipeline quality across restarts.
"""

from __future__ import annotations

import sqlite3
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Default SQLite database path for metrics persistence
_METRICS_DB_PATH = Path("data/metrics.db")


@dataclass
class QueryMetricPoint:
    """A single metric data point recorded for a query.

    Attributes:
        timestamp: Unix timestamp of the recording.
        latency_ms: Query processing latency in milliseconds.
        confidence: Model confidence score (0-1).
        num_docs_retrieved: Number of documents retrieved.
        num_docs_relevant: Number of documents judged relevant.
        provider: LLM provider used.
        model: Model identifier used.
        rbac_blocked: Whether the query was blocked by RBAC.
    """

    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    confidence: float = 0.0
    num_docs_retrieved: int = 0
    num_docs_relevant: int = 0
    provider: str = "ollama"
    model: str = ""
    rbac_blocked: bool = False


class MetricsCollector:
    """Metrics collector with in-memory cache and optional SQLite persistence.

    Records query-level metrics and provides statistical aggregations
    for latency, confidence, retrieval quality, RBAC blocking rates,
    and provider comparisons. Metrics survive application restarts when
    SQLite persistence is enabled.
    """

    def __init__(self, persist_to_sqlite: bool = True) -> None:
        """Initialize the metrics collector.

        Args:
            persist_to_sqlite: Whether to persist metrics to SQLite.
        """
        self._data_points: list[QueryMetricPoint] = []
        self._persist = persist_to_sqlite
        self._db_path = _METRICS_DB_PATH

        if self._persist:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._init_sqlite()
            self._load_from_sqlite()

    def _init_sqlite(self) -> None:
        """Create the metrics table if it doesn't exist."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        latency_ms REAL NOT NULL,
                        confidence REAL NOT NULL,
                        num_docs_retrieved INTEGER NOT NULL,
                        num_docs_relevant INTEGER NOT NULL,
                        provider TEXT NOT NULL,
                        model TEXT NOT NULL,
                        rbac_blocked INTEGER NOT NULL DEFAULT 0
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
                    ON query_metrics(timestamp)
                """)
                conn.commit()
        except Exception as exc:
            logger.warning("metrics_sqlite_init_failed", error=str(exc))
            self._persist = False

    def _load_from_sqlite(self) -> None:
        """Load recent metrics from SQLite into memory."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                # Load last 7 days of metrics
                cutoff = time.time() - (7 * 24 * 3600)
                cursor = conn.execute(
                    """
                    SELECT timestamp, latency_ms, confidence, num_docs_retrieved,
                           num_docs_relevant, provider, model, rbac_blocked
                    FROM query_metrics
                    WHERE timestamp > ?
                    ORDER BY timestamp DESC
                    LIMIT 10000
                    """,
                    (cutoff,),
                )
                rows = cursor.fetchall()
                for row in rows:
                    self._data_points.append(
                        QueryMetricPoint(
                            timestamp=row[0],
                            latency_ms=row[1],
                            confidence=row[2],
                            num_docs_retrieved=row[3],
                            num_docs_relevant=row[4],
                            provider=row[5],
                            model=row[6],
                            rbac_blocked=bool(row[7]),
                        )
                    )
                logger.info("metrics_loaded_from_sqlite", count=len(self._data_points))
        except Exception as exc:
            logger.warning("metrics_sqlite_load_failed", error=str(exc))

    def _save_to_sqlite(self, point: QueryMetricPoint) -> None:
        """Save a single metric point to SQLite.

        Args:
            point: The metric point to persist.
        """
        if not self._persist:
            return
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO query_metrics
                    (timestamp, latency_ms, confidence, num_docs_retrieved,
                     num_docs_relevant, provider, model, rbac_blocked)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        point.timestamp,
                        point.latency_ms,
                        point.confidence,
                        point.num_docs_retrieved,
                        point.num_docs_relevant,
                        point.provider,
                        point.model,
                        1 if point.rbac_blocked else 0,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.debug("metrics_sqlite_save_failed", error=str(exc))

    def record_query(
        self,
        latency_ms: float,
        confidence: float,
        num_docs_retrieved: int,
        num_docs_relevant: int,
        provider: str,
        model: str,
        rbac_blocked: bool = False,
    ) -> None:
        """Record metrics for a single query execution.

        Args:
            latency_ms: Query processing time in milliseconds.
            confidence: Model confidence score (0.0 to 1.0).
            num_docs_retrieved: Total documents returned by retrieval.
            num_docs_relevant: Documents judged relevant by grader.
            provider: LLM provider name (e.g., "ollama", "groq").
            model: Model identifier used for generation.
            rbac_blocked: Whether the query was blocked by RBAC filters.
        """
        point = QueryMetricPoint(
            latency_ms=latency_ms,
            confidence=confidence,
            num_docs_retrieved=num_docs_retrieved,
            num_docs_relevant=num_docs_relevant,
            provider=provider,
            model=model,
            rbac_blocked=rbac_blocked,
        )
        self._data_points.append(point)

    def get_data_points(self) -> list[QueryMetricPoint]:
        """Return a copy of all collected data points."""
        return list(self._data_points)

    def get_latency_stats(self) -> dict[str, float]:
        """Compute latency percentile statistics.

        Returns:
            Dictionary with p50, p90, p99, mean, min, max latency values.
            Returns empty dict if no data points exist.
        """
        latencies = [p.latency_ms for p in self._data_points if not p.rbac_blocked]
        if not latencies:
            return {}

        sorted_lat = sorted(latencies)
        n = len(sorted_lat)

        return {
            "p50": sorted_lat[int(n * 0.50)] if n > 0 else 0.0,
            "p90": sorted_lat[int(n * 0.90)] if n > 0 else 0.0,
            "p99": sorted_lat[min(int(n * 0.99), n - 1)] if n > 0 else 0.0,
            "mean": statistics.mean(sorted_lat),
            "min": min(sorted_lat),
            "max": max(sorted_lat),
            "count": float(n),
        }

    def get_confidence_stats(self) -> dict[str, float]:
        """Compute confidence score statistics.

        Returns:
            Dictionary with mean, median, std, and distribution info.
            Returns empty dict if no data points exist.
        """
        scores = [p.confidence for p in self._data_points if not p.rbac_blocked]
        if not scores:
            return {}

        result: dict[str, float] = {
            "mean": statistics.mean(scores),
            "median": statistics.median(scores),
            "min": min(scores),
            "max": max(scores),
            "count": float(len(scores)),
        }

        if len(scores) >= 2:
            result["std"] = statistics.stdev(scores)
        else:
            result["std"] = 0.0

        # Distribution buckets
        result["low_confidence_rate"] = sum(1 for s in scores if s < 0.5) / len(scores)
        result["medium_confidence_rate"] = sum(1 for s in scores if 0.5 <= s < 0.8) / len(scores)
        result["high_confidence_rate"] = sum(1 for s in scores if s >= 0.8) / len(scores)

        return result

    def get_retrieval_stats(self) -> dict[str, float]:
        """Compute retrieval quality statistics.

        Returns:
            Dictionary with average relevance ratio, docs retrieved, docs relevant.
            Returns empty dict if no data points exist.
        """
        points = [p for p in self._data_points if not p.rbac_blocked]
        if not points:
            return {}

        relevance_ratios = [
            p.num_docs_relevant / p.num_docs_retrieved if p.num_docs_retrieved > 0 else 0.0
            for p in points
        ]

        return {
            "avg_relevance_ratio": statistics.mean(relevance_ratios),
            "avg_docs_retrieved": statistics.mean([p.num_docs_retrieved for p in points]),
            "avg_docs_relevant": statistics.mean([p.num_docs_relevant for p in points]),
            "total_queries": float(len(points)),
            "zero_result_rate": sum(1 for p in points if p.num_docs_retrieved == 0) / len(points),
        }

    def get_rbac_stats(self) -> dict[str, float]:
        """Compute RBAC blocking statistics.

        Returns:
            Dictionary with total blocks, block rate, and total queries.
            Returns empty dict if no data points exist.
        """
        if not self._data_points:
            return {}

        total = len(self._data_points)
        blocked = sum(1 for p in self._data_points if p.rbac_blocked)

        return {
            "total_queries": float(total),
            "total_blocks": float(blocked),
            "block_rate": blocked / total if total > 0 else 0.0,
        }

    def get_provider_comparison(self) -> dict[str, dict[str, float]]:
        """Compute statistics grouped by LLM provider.

        Returns:
            Dictionary keyed by provider name, each containing latency
            and confidence stats for that provider.
        """
        providers: dict[str, list[QueryMetricPoint]] = {}
        for point in self._data_points:
            if point.rbac_blocked:
                continue
            if point.provider not in providers:
                providers[point.provider] = []
            providers[point.provider].append(point)

        result: dict[str, dict[str, float]] = {}
        for provider, points in providers.items():
            latencies = [p.latency_ms for p in points]
            confidences = [p.confidence for p in points]
            result[provider] = {
                "count": float(len(points)),
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0.0,
                "avg_confidence": statistics.mean(confidences) if confidences else 0.0,
                "p90_latency_ms": sorted(latencies)[int(len(latencies) * 0.9)]
                if latencies
                else 0.0,
            }

        return result

    def get_summary(self) -> dict[str, Any]:
        """Get comprehensive summary combining all metric categories.

        Returns:
            Dictionary with all statistics combined under descriptive keys.
        """
        return {
            "latency": self.get_latency_stats(),
            "confidence": self.get_confidence_stats(),
            "retrieval": self.get_retrieval_stats(),
            "rbac": self.get_rbac_stats(),
            "providers": self.get_provider_comparison(),
            "total_data_points": len(self._data_points),
        }

    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all data points as a Pandas DataFrame.

        Returns:
            DataFrame with one row per recorded query metric point.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for DataFrame export. Install with: pip install pandas"
            ) from exc

        records = [
            {
                "timestamp": p.timestamp,
                "latency_ms": p.latency_ms,
                "confidence": p.confidence,
                "num_docs_retrieved": p.num_docs_retrieved,
                "num_docs_relevant": p.num_docs_relevant,
                "provider": p.provider,
                "model": p.model,
                "rbac_blocked": p.rbac_blocked,
            }
            for p in self._data_points
        ]
        return pd.DataFrame(records)

    def reset(self) -> None:
        """Clear all recorded data points."""
        self._data_points.clear()

    @property
    def count(self) -> int:
        """Return the total number of recorded data points."""
        return len(self._data_points)


# Module-level singleton instance
metrics_collector = MetricsCollector()
