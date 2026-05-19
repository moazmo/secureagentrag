"""Verify the integrity of the tamper-evident audit log chain.

Usage:
    uv run python -m scripts.verify_audit_chain
    uv run python -m scripts.verify_audit_chain --start 2026-05-01 --end 2026-05-19
    uv run python -m scripts.verify_audit_chain --log-dir /var/log/secureagentrag

Exit codes:
    0 - chain valid
    1 - chain broken (tampering detected)
    2 - audit directory missing / unreadable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from utils.audit import AuditLogger


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log-dir", default=None, help="Audit log directory (defaults to settings)"
    )
    parser.add_argument("--start", default=None, help="ISO start date (inclusive)")
    parser.add_argument("--end", default=None, help="ISO end date (inclusive)")
    args = parser.parse_args()

    logger = AuditLogger(log_dir=args.log_dir)

    if not Path(logger._log_dir).exists():
        print(f"ERROR: audit log dir not found: {logger._log_dir}", file=sys.stderr)
        return 2

    result = logger.verify_chain(start_date=args.start, end_date=args.end)

    if result["valid"]:
        print(f"OK  chain valid  entries_checked={result['checked']}")
        print(f"    last_hash={result['last_hash']}")
        return 0

    print(f"FAIL chain broken  entries_checked={result['checked']}", file=sys.stderr)
    for reason in result["broken_at"]:
        print(f"     {reason}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
