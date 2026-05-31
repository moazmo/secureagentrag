"""Structured audit logging for compliance and security tracking.

Records all sensitive operations (queries, data access, ingestion events,
security blocks) with structured metadata. Persists entries to daily JSONL
files for later review, export to SIEM systems, or compliance reporting.

Each entry carries a SHA-256 ``entry_hash`` and the ``prev_hash`` of the
previous entry, forming a hash chain. Any in-place edit, deletion, or
re-ordering of past entries breaks the chain and is detected by
``verify_chain`` / ``scripts/verify_audit_chain.py``.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config.settings import settings
from utils.logging import get_logger
from utils.pii import redact_dict

_audit_log = get_logger("audit")

GENESIS_HASH = "GENESIS"


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
        prev_hash: SHA-256 of the previous entry's ``entry_hash`` (``GENESIS`` for first).
        entry_hash: SHA-256 over canonical JSON of this entry excluding ``entry_hash``.
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
    prev_hash: str = ""
    entry_hash: str = ""

    def compute_hash(self) -> str:
        """Hash the canonical JSON of this entry (excluding ``entry_hash``).

        SHA-256 by default (tamper-evident). When ``settings.audit_hmac_key`` is
        set the digest is HMAC-SHA256 keyed by that secret (tamper-resistant) —
        ``verify_chain`` recomputes the same way, so flipping the key on a fresh
        chain upgrades the integrity guarantee with no other code change.
        """
        payload = self.model_dump(mode="json", exclude={"entry_hash"})
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        key = settings.audit_hmac_key
        if key:
            return hmac.new(key.encode("utf-8"), canonical, hashlib.sha256).hexdigest()
        return hashlib.sha256(canonical).hexdigest()


class AuditLogger:
    """Structured audit trail for security-sensitive operations.

    Persists entries to daily JSONL files and emits structured log events
    that can be collected by log aggregators for compliance reporting.

    Args:
        log_dir: Directory path for audit log files. Created if not exists.
    """

    def __init__(self, log_dir: str | None = None) -> None:
        """Initialize the audit logger with a target log directory.

        Args:
            log_dir: Directory path for storing daily JSONL audit files.
                Defaults to ``settings.audit_log_dir`` so deployments can pin
                an absolute path via the ``SAR_AUDIT_LOG_DIR`` env var.
        """
        self._log_dir = Path(log_dir if log_dir is not None else settings.audit_log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        # In-process mutex for hash-chain integrity. Multi-process deployments
        # should serialise audit writes (single writer) or migrate to Postgres.
        self._chain_lock = threading.Lock()
        self._last_hash: str | None = None  # Lazily resolved on first write

    def _read_last_hash(self) -> str:
        """Scan audit directory to find the most recent entry's hash.

        Used once on first write to bootstrap the chain. Returns ``GENESIS``
        if no prior entries exist.
        """
        try:
            files = sorted(self._log_dir.glob("audit_*.jsonl"))
            for path in reversed(files):
                # Walk file backwards line by line — newest entry is last
                with open(path, encoding="utf-8") as f:
                    lines = [ln for ln in f if ln.strip()]
                if not lines:
                    continue
                last = json.loads(lines[-1])
                last_hash = last.get("entry_hash")
                if last_hash:
                    return last_hash
            return GENESIS_HASH
        except Exception as exc:
            _audit_log.warning("audit_chain_bootstrap_failed", error=str(exc))
            return GENESIS_HASH

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

    def log_feedback(
        self,
        *,
        user_id: str,
        org_id: str = "",
        rating: str,
        query: str = "",
        **kwargs: Any,
    ) -> AuditEntry:
        """Record a user thumbs-up/down on an answer.

        Lands on the same SHA-256 hash chain as every other event (PII-redacted
        before persistence), so feedback is itself tamper-evident and exportable
        via ``/byok/audit``. ``rating`` is normalised to ``up`` / ``down``.

        Args:
            user_id: Identifier of the user giving feedback.
            org_id: Organization identifier.
            rating: ``up`` or ``down``.
            query: The question the rated answer responded to.
            **kwargs: Extra metadata (e.g. answer_summary).

        Returns:
            The persisted AuditEntry.
        """
        entry = AuditEntry(
            action="feedback",
            user_id=user_id,
            org_id=org_id,
            details={"rating": rating, "query": query},
            status="success",
            metadata=kwargs,
        )
        self._persist(entry)
        _audit_log.info("feedback_recorded", user_id=user_id, rating=rating)
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
        """Append an audit entry to the daily JSONL file with hash chaining.

        File naming convention: ``audit_YYYY-MM-DD.jsonl``. Each entry's
        ``prev_hash`` references the previous entry's ``entry_hash`` so the
        whole stream is a tamper-evident chain. Writes are serialised by
        ``self._chain_lock`` to prevent concurrent re-use of a ``prev_hash``.

        Args:
            entry: The AuditEntry to persist (hash fields populated here).
        """
        try:
            with self._chain_lock:
                if self._last_hash is None:
                    self._last_hash = self._read_last_hash()
                # Scrub PII *before* hashing so the on-disk entry and its
                # signature match. Live in-memory state is unaffected because
                # _persist is the boundary to durable storage.
                entry.details = redact_dict(entry.details)
                entry.metadata = redact_dict(entry.metadata)
                entry.prev_hash = self._last_hash
                entry.entry_hash = entry.compute_hash()

                today = date.today().isoformat()
                file_path = self._log_dir / f"audit_{today}.jsonl"
                line = entry.model_dump_json() + "\n"
                with open(file_path, "a", encoding="utf-8") as f:
                    f.write(line)

                self._last_hash = entry.entry_hash
        except Exception as exc:
            _audit_log.error("audit_persist_failed", error=str(exc))

    def verify_chain(
        self,
        start_date: date | str | None = None,
        end_date: date | str | None = None,
    ) -> dict[str, Any]:
        """Verify the SHA-256 hash chain across one or more daily audit files.

        Walks every entry in ``[start_date, end_date]`` (or all files when
        both are ``None``) and recomputes ``entry_hash`` while checking that
        each entry's ``prev_hash`` matches the previous entry's hash.

        Args:
            start_date: First date (inclusive). ``None`` for "earliest".
            end_date: Last date (inclusive). ``None`` for "latest".

        Returns:
            Dict with keys ``valid`` (bool), ``checked`` (int), ``broken_at``
            (list of file:line:reason strings), and ``last_hash`` (str).
        """
        files = sorted(self._log_dir.glob("audit_*.jsonl"))
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = date.fromisoformat(start_date)
            files = [p for p in files if date.fromisoformat(p.stem.split("_", 1)[1]) >= start_date]
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = date.fromisoformat(end_date)
            files = [p for p in files if date.fromisoformat(p.stem.split("_", 1)[1]) <= end_date]

        broken: list[str] = []
        checked = 0
        expected_prev = GENESIS_HASH if start_date is None else None
        last_hash = GENESIS_HASH

        for path in files:
            with open(path, encoding="utf-8") as f:
                for lineno, raw in enumerate(f, start=1):
                    raw = raw.strip()
                    if not raw:
                        continue
                    try:
                        entry = AuditEntry.model_validate_json(raw)
                    except Exception as exc:
                        broken.append(f"{path.name}:{lineno}:parse_error:{exc}")
                        return {
                            "valid": False,
                            "checked": checked,
                            "broken_at": broken,
                            "last_hash": last_hash,
                        }

                    recomputed = entry.compute_hash()
                    if recomputed != entry.entry_hash:
                        broken.append(
                            f"{path.name}:{lineno}:hash_mismatch:"
                            f"stored={entry.entry_hash[:12]} recomputed={recomputed[:12]}"
                        )

                    if expected_prev is not None and entry.prev_hash != expected_prev:
                        broken.append(
                            f"{path.name}:{lineno}:chain_broken:"
                            f"prev_hash={entry.prev_hash[:12]} expected={expected_prev[:12]}"
                        )

                    expected_prev = entry.entry_hash
                    last_hash = entry.entry_hash
                    checked += 1

        return {
            "valid": not broken,
            "checked": checked,
            "broken_at": broken,
            "last_hash": last_hash,
        }

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
