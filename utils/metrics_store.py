"""Persistent metrics storage using SQLite for evaluation data.

Survives Streamlit reruns and app restarts, enabling long-term performance
tracking and trend analysis across sessions.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from utils.logging import get_logger

logger = get_logger(__name__)

# Default database path
DEFAULT_DB_PATH = Path("data/metrics.db")

# Thread-local connection storage
_local = threading.local()


def _get_db_path() -> Path:
    """Return the metrics database path."""
    return DEFAULT_DB_PATH


@contextmanager
def _get_connection():
    """Get a thread-local SQLite connection with proper cleanup."""
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _init_schema() -> None:
    """Initialize the metrics database schema if not present."""
    with _get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                query TEXT NOT NULL,
                confidence REAL,
                latency_ms REAL,
                query_type TEXT,
                user_id TEXT,
                model TEXT,
                security_passed INTEGER DEFAULT 1,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_timestamp
            ON evaluation_metrics(timestamp)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_user
            ON evaluation_metrics(user_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_metrics_query_type
            ON evaluation_metrics(query_type)
            """
        )
        conn.commit()


# Initialize schema on module load
_init_schema()


def store_metric(
    query: str,
    confidence: float | None = None,
    latency_ms: float | None = None,
    query_type: str | None = None,
    user_id: str | None = None,
    model: str | None = None,
    security_passed: bool = True,
    metadata: dict[str, Any] | None = None,
) -> int:
    """Store a single evaluation metric in SQLite.

    Args:
        query: The user's query text.
        confidence: Confidence score (0.0-1.0).
        latency_ms: Query processing latency in milliseconds.
        query_type: Classification of the query type.
        user_id: Identifier of the querying user.
        model: Model/provider used for the response.
        security_passed: Whether security checks passed.
        metadata: Additional JSON-serializable metadata.

    Returns:
        The ID of the inserted row.
    """
    timestamp = datetime.now(UTC).isoformat()
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO evaluation_metrics
            (timestamp, query, confidence, latency_ms, query_type, user_id, model, security_passed, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                query,
                confidence,
                latency_ms,
                query_type,
                user_id,
                model,
                1 if security_passed else 0,
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()
        row_id = cursor.lastrowid or 0
        logger.debug("metric_stored", row_id=row_id, query_len=len(query))
        return row_id


def get_metrics(
    limit: int = 1000,
    user_id: str | None = None,
    since: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve evaluation metrics from persistent storage.

    Args:
        limit: Maximum number of records to return.
        user_id: Optional filter by user.
        since: Optional ISO timestamp to filter records after.

    Returns:
        List of metric dictionaries.
    """
    with _get_connection() as conn:
        query = "SELECT * FROM evaluation_metrics WHERE 1=1"
        params: list[Any] = []

        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        if since:
            query += " AND timestamp > ?"
            params.append(since)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()

        results = []
        for row in rows:
            row_dict = dict(row)
            if row_dict.get("metadata"):
                with suppress(json.JSONDecodeError):
                    row_dict["metadata"] = json.loads(row_dict["metadata"])
            row_dict["security_passed"] = bool(row_dict.get("security_passed", 1))
            results.append(row_dict)

        return results


def get_metrics_summary() -> dict[str, Any]:
    """Get aggregate summary statistics from stored metrics.

    Returns:
        Dict with total_queries, avg_confidence, avg_latency_ms, blocked_count.
    """
    with _get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) as total,
                AVG(confidence) as avg_confidence,
                AVG(latency_ms) as avg_latency,
                SUM(CASE WHEN security_passed = 0 THEN 1 ELSE 0 END) as blocked
            FROM evaluation_metrics
            """
        ).fetchone()

        if row is None:
            return {
                "total_queries": 0,
                "avg_confidence": 0.0,
                "avg_latency_ms": 0.0,
                "blocked_count": 0,
            }

        return {
            "total_queries": row["total"] or 0,
            "avg_confidence": row["avg_confidence"] or 0.0,
            "avg_latency_ms": row["avg_latency"] or 0.0,
            "blocked_count": row["blocked"] or 0,
        }


def cleanup_old_metrics(days: int = 90) -> int:
    """Delete metrics older than the specified number of days.

    Args:
        days: Retention period in days. Default 90.

    Returns:
        Number of rows deleted.
    """
    # SQLite doesn't have direct date arithmetic; use string comparison
    # For proper cleanup, we'd need to parse, but this is a best-effort
    # approach using a simple time-based heuristic
    cutoff_timestamp = time.time() - (days * 86400)
    cutoff_iso = datetime.fromtimestamp(cutoff_timestamp, UTC).isoformat()

    with _get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM evaluation_metrics WHERE timestamp < ?",
            (cutoff_iso,),
        )
        conn.commit()
        deleted = cursor.rowcount
        logger.info("metrics_cleaned_up", deleted=deleted, retention_days=days)
        return deleted


def sync_session_metrics(session_data: list[dict[str, Any]]) -> int:
    """Sync in-memory session metrics to persistent storage.

    Args:
        session_data: List of metric dicts from st.session_state.

    Returns:
        Number of metrics synced.
    """
    if not session_data:
        return 0

    count = 0
    for item in session_data:
        try:
            store_metric(
                query=item.get("query", ""),
                confidence=item.get("confidence"),
                latency_ms=item.get("latency_ms"),
                query_type=item.get("query_type"),
                user_id=item.get("user"),
                model=item.get("model"),
                security_passed=item.get("security_passed", True),
                metadata={k: v for k, v in item.items() if k not in {
                    "query", "confidence", "latency_ms", "query_type",
                    "user", "model", "security_passed", "timestamp"
                }},
            )
            count += 1
        except Exception as exc:
            logger.warning("metric_sync_failed", error=str(exc))

    logger.info("session_metrics_synced", count=count)
    return count
