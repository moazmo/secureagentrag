"""Structured audit logging for compliance and security tracking.

Records all sensitive operations (queries, data access, ingestion events,
security blocks) with structured metadata. Persists entries to daily JSONL
files for later review, export to SIEM systems, or compliance reporting.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from utils.logging import get_logger

_audit_log = get_logger("audit")


class AuditEntry(BaseModel):
    """A single structured audit log entry.

    Attributes:
        timestamp: UTC timestamp of the event.
        action: Event category — "query", "upload", "access", "security_block", "inference".
        user_id: Identifier of the user who triggered the event.
        org_id: Organization identifier for multi-tenant tracking.
        details: Action-specific details (query text, file path, etc.).
        sensitivity_level: Data sensitivity classification (low, medium, high).
        status: Outcome — "success", "blocked", or "error".
        latency_ms: Operation latency in milliseconds (if applicable).
        metadata: Additional unstructured metadata.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    action: str
    user_id: str
    org_id: str = ""
    details: dict[str, Any] = Field(default_factory=dict)
    sensitivity_level: str = "low"
    status: str = "success"
    latency_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditLogger:
    """Structured audit trail for security-sensitive operations.

    Persists entries to daily JSONL files and emits structured log events
    that can be collected by log aggregators for compliance reporting.

    Args:
        log_dir: Directory path for audit log files. Created if not exists.
    """

    def __init__(self, log_dir: str = "audit_logs") -> None:
        """Initialize the audit logger with a target log directory.

        Args:
            log_dir: Directory path for storing daily JSONL audit files.
        """
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)

    def log_query(
        self,
        *,
        user_id: str,
        org_id: str = "",
        query: str,
        response_summary: str = "",
        sensitivity: str = "low",
        status: str = "success",
        latency_ms: float | None = None,
        **kwargs: Any,
    ) -> AuditEntry:
        """Record a user query event.

        Args:
            user_id: Identifier of the user making the query.
            org_id: Organization identifier.
            query: The natural-language query text.
            response_summary: Brief summary of the generated response.
            sensitivity: Data sensitivity level.
            status: Query outcome status.
            latency_ms: Query processing time in milliseconds.
            **kwargs: Additional metadata fields.

        Returns:
            The persisted AuditEntry.
        """
        entry = AuditEntry(
            action="query",
            user_id=user_id,
            org_id=org_id,
            details={"query": query, "response_summary": response_summary},
            sensitivity_level=sensitivity,
            status=status,
            latency_ms=latency_ms,
            metadata=kwargs,
        )
        self._persist(entry)
        _audit_log.info(
            "query_executed",
            user_id=user_id,
            query_length=len(query),
            status=status,
            latency_ms=latency_ms,
        )
        return entry

    def log_access(
        self,
        *,
        user_id: str,
        org_id: str = "",
        documents_accessed: list[str] | None = None,
        sensitivity: str = "low",
        **kwargs: Any,
    ) -> AuditEntry:
        """Record a document access event.

        Args:
            user_id: Identifier of the requesting user.
            org_id: Organization identifier.
            documents_accessed: List of document IDs or names accessed.
            sensitivity: Data sensitivity level.
            **kwargs: Additional metadata fields.

        Returns:
            The persisted AuditEntry.
        """
        entry = AuditEntry(
            action="access",
            user_id=user_id,
            org_id=org_id,
            details={"documents_accessed": documents_accessed or []},
            sensitivity_level=sensitivity,
            status="success",
            metadata=kwargs,
        )
        self._persist(entry)
        _audit_log.info(
            "access_event",
            user_id=user_id,
            doc_count=len(documents_accessed or []),
            sensitivity=sensitivity,
        )
        return entry

    def log_ingestion(
        self,
        *,
        user_id: str,
        org_id: str = "",
        file_path: str = "",
        num_chunks: int = 0,
        status: str = "success",
        # Legacy support for existing pipeline calls
        document_name: str = "",
        chunk_count: int = 0,
        **kwargs: Any,
    ) -> AuditEntry:
        """Record a document ingestion event.

        Args:
            user_id: Identifier of the user who triggered ingestion.
            org_id: Organization identifier.
            file_path: Path of the ingested document.
            num_chunks: Number of chunks produced.
            status: Ingestion outcome status.
            document_name: Legacy parameter (alias for file_path).
            chunk_count: Legacy parameter (alias for num_chunks).
            **kwargs: Additional metadata fields.

        Returns:
            The persisted AuditEntry.
        """
        # Support legacy call signature from pipeline.py
        actual_path = file_path or document_name
        actual_chunks = num_chunks or chunk_count

        entry = AuditEntry(
            action="upload",
            user_id=user_id,
            org_id=org_id,
            details={"file_path": actual_path, "num_chunks": actual_chunks},
            sensitivity_level=kwargs.pop("sensitivity", "low"),
            status=status,
            metadata=kwargs,
        )
        self._persist(entry)
        _audit_log.info(
            "document_ingested",
            user_id=user_id,
            file_path=actual_path,
            chunk_count=actual_chunks,
            status=status,
        )
        return entry

    def log_security_event(
        self,
        *,
        user_id: str,
        org_id: str = "",
        event_type: str,
        details: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AuditEntry:
        """Record a security-relevant event (block, escalation, anomaly).

        Args:
            user_id: Identifier of the user involved.
            org_id: Organization identifier.
            event_type: Type of security event (e.g., "rbac_block", "injection_attempt").
            details: Event-specific details.
            **kwargs: Additional metadata fields.

        Returns:
            The persisted AuditEntry.
        """
        entry = AuditEntry(
            action="security_block",
            user_id=user_id,
            org_id=org_id,
            details={"event_type": event_type, **(details or {})},
            sensitivity_level="high",
            status="blocked",
            metadata=kwargs,
        )
        self._persist(entry)
        _audit_log.warning(
            "security_event",
            user_id=user_id,
            event_type=event_type,
            org_id=org_id,
        )
        return entry

    def log_access_legacy(
        self,
        *,
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record a resource access event (legacy interface).

        Maintains backward compatibility with the original AuditLogger API.

        Args:
            user_id: Identifier of the requesting user.
            resource: The resource being accessed.
            action: Action attempted (read, write, delete).
            granted: Whether access was granted.
            metadata: Additional context.
        """
        entry = AuditEntry(
            action="access",
            user_id=user_id,
            org_id="",
            details={"resource": resource, "action": action, "granted": granted},
            sensitivity_level="medium",
            status="success" if granted else "blocked",
            metadata=metadata or {},
        )
        self._persist(entry)
        _audit_log.info(
            "access_event",
            user_id=user_id,
            resource=resource,
            action=action,
            granted=granted,
        )

    def _persist(self, entry: AuditEntry) -> None:
        """Append an audit entry to the daily JSONL file.

        File naming convention: ``audit_YYYY-MM-DD.jsonl``

        Args:
            entry: The AuditEntry to persist.
        """
        try:
            today = date.today().isoformat()
            file_path = self._log_dir / f"audit_{today}.jsonl"
            line = entry.model_dump_json() + "\n"
            with open(file_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as exc:
            _audit_log.error("audit_persist_failed", error=str(exc))

    def get_entries(
        self,
        start_date: date | str,
        end_date: date | str,
        user_id: str | None = None,
        action: str | None = None,
    ) -> list[AuditEntry]:
        """Read and filter audit entries from persisted JSONL files.

        Args:
            start_date: Start date (inclusive) for the query range.
            end_date: End date (inclusive) for the query range.
            user_id: Optional filter by user identifier.
            action: Optional filter by action type.

        Returns:
            List of matching AuditEntry objects.
        """
        if isinstance(start_date, str):
            start_date = date.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = date.fromisoformat(end_date)

        entries: list[AuditEntry] = []
        current = start_date

        while current <= end_date:
            file_path = self._log_dir / f"audit_{current.isoformat()}.jsonl"
            if file_path.exists():
                try:
                    with open(file_path, encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            entry = AuditEntry.model_validate_json(line)
                            # Apply filters
                            if user_id and entry.user_id != user_id:
                                continue
                            if action and entry.action != action:
                                continue
                            entries.append(entry)
                except Exception as exc:
                    _audit_log.error(
                        "audit_read_failed",
                        file=str(file_path),
                        error=str(exc),
                    )
            # Advance to next day
            from datetime import timedelta

            current = current + timedelta(days=1)

        return entries

    def get_summary(
        self,
        start_date: date | str,
        end_date: date | str,
    ) -> dict[str, Any]:
        """Generate aggregate summary of audit entries over a date range.

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).

        Returns:
            Dictionary with counts grouped by action, user, and status.
        """
        entries = self.get_entries(start_date, end_date)

        by_action: dict[str, int] = {}
        by_user: dict[str, int] = {}
        by_status: dict[str, int] = {}

        for entry in entries:
            by_action[entry.action] = by_action.get(entry.action, 0) + 1
            by_user[entry.user_id] = by_user.get(entry.user_id, 0) + 1
            by_status[entry.status] = by_status.get(entry.status, 0) + 1

        return {
            "total_entries": len(entries),
            "by_action": by_action,
            "by_user": by_user,
            "by_status": by_status,
            "date_range": {
                "start": str(start_date),
                "end": str(end_date),
            },
        }


# Module-level singleton
audit_logger = AuditLogger()
