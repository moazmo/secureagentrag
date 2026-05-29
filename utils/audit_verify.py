"""Scheduled SHA-256 audit hash-chain verification.

The audit log is tamper-evident (each JSONL entry chains ``prev_hash`` →
``entry_hash``), but tamper-*evidence* only matters if someone actually looks.
``scripts/verify_audit_chain.py`` does it on demand; this module runs the same
``AuditLogger.verify_chain()`` periodically from the FastAPI lifespan so a
broken chain is surfaced (log + Prometheus metric) without a human in the loop.

Reads local JSONL only — no external services — so it is safe to run anywhere,
including the public BYOK Space.
"""

from __future__ import annotations

from typing import Any

from config.settings import settings
from utils.audit import audit_logger
from utils.logging import get_logger
from utils.metrics import record_audit_verification

logger = get_logger(__name__)


def run_audit_verification(audit: Any | None = None) -> dict[str, Any]:
    """Verify the audit hash chain once and emit a log line + metric.

    Returns the ``verify_chain`` result dict (``valid`` / ``checked`` /
    ``broken_at`` / ``last_hash``). Never raises — a verification failure is
    reported, not propagated, so the scheduler keeps running.
    """
    al = audit or audit_logger
    try:
        result = al.verify_chain()
    except Exception as exc:
        logger.error("audit_verify_error", error=str(exc))
        record_audit_verification("error", valid=False)
        return {"valid": False, "checked": 0, "broken_at": [f"error:{exc}"], "last_hash": ""}

    if result.get("valid"):
        logger.info("audit_verify_ok", checked=result.get("checked", 0))
        record_audit_verification("valid", valid=True)
    else:
        # broken_at is a list of "file:line:reason" strings — safe to log
        # (no PII; the audit entries themselves are already PII-redacted).
        logger.error(
            "audit_verify_tamper_detected",
            checked=result.get("checked", 0),
            broken_at=result.get("broken_at", []),
        )
        record_audit_verification("broken", valid=False)
    return result


def schedule_audit_verification(
    *, interval_hours: int | None = None, audit: Any | None = None
) -> Any | None:
    """Start an APScheduler job that runs :func:`run_audit_verification`.

    Called from the FastAPI ``lifespan``. Returns the ``AsyncIOScheduler``
    instance, or ``None`` when disabled or when APScheduler is unavailable (in
    which case a single startup sweep still runs so each boot verifies once).
    """
    if not settings.audit_verify_enabled:
        logger.debug("audit_verify_not_scheduled", reason="disabled")
        return None

    interval = interval_hours or settings.audit_verify_interval_hours

    try:
        from apscheduler.schedulers.asyncio import (
            AsyncIOScheduler,  # type: ignore[import-not-found]
        )
    except ImportError:
        logger.warning("apscheduler_missing", action="single-shot audit verify instead")
        run_audit_verification(audit)
        return None

    # Verify once at startup, then on the interval.
    run_audit_verification(audit)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_audit_verification,
        "interval",
        hours=interval,
        args=[audit],
        id="audit-chain-verify",
        replace_existing=True,
    )
    scheduler.start()
    logger.info("audit_verify_scheduled", every_hours=interval)
    return scheduler
