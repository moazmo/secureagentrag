"""Hello-world FastAPI for SecureAgentRAG HF Space smoke test.

Verifies:
- HF Docker SDK builds a custom image
- Free CPU Basic (2 vCPU / 16 GB) accepts our deps
- Port 7860 reachable from public internet (Egypt origin)

This is a smoke test only. Real backend lands in Phase 2 (see
launch-plan/03-backend-byok.md).
"""

from __future__ import annotations

import os
import time

from fastapi import FastAPI

app = FastAPI(title="SecureAgentRAG API — smoke", version="0.0.1-smoke")

_BOOT_TIME = time.monotonic()


@app.get("/")
def root() -> dict[str, object]:
    return {
        "service": "secureagentrag-api",
        "phase": "1-smoke",
        "ok": True,
        "uptime_seconds": round(time.monotonic() - _BOOT_TIME, 2),
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "service": "secureagentrag-api",
        "phase": "1-smoke",
        "python": os.environ.get("PYTHON_VERSION", "unknown"),
        "uptime_seconds": round(time.monotonic() - _BOOT_TIME, 2),
    }
