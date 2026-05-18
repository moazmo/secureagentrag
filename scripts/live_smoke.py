"""Live single-query smoke test against the running stack.

Usage:
    uv run python -m scripts.live_smoke "your query"

Prints per-phase latency in real time and the final answer.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.graph import run_rag_pipeline_stream  # noqa: E402
from ingestion.metadata import UserContext  # noqa: E402


async def main() -> int:
    query = sys.argv[1] if len(sys.argv) > 1 else "What are the data classification levels?"

    user = UserContext(
        user_id="admin_01",
        org_id="acme_corp",
        roles=["admin", "analyst", "viewer"],
        clearance_level=3,
    )

    t0 = time.time()
    tokens: list[str] = []
    final = None

    print(f"Query: {query!r}", flush=True)
    print(f"User:  {user.user_id} (clearance {user.clearance_level})", flush=True)
    print("-" * 60, flush=True)

    async for ev in run_rag_pipeline_stream(query=query, user_context=user, thread_id="live"):
        etype = ev["type"]
        if etype == "phase":
            print(f"  [{time.time() - t0:5.1f}s] phase: {ev['name']}", flush=True)
        elif etype == "token":
            tokens.append(ev["text"])
        elif etype == "blocked":
            print(f"  BLOCKED: {ev['message']}", flush=True)
            return 1
        elif etype == "final":
            final = ev
            print(f"  [{time.time() - t0:5.1f}s] final state ready", flush=True)

    elapsed = time.time() - t0
    text = "".join(tokens)

    print("-" * 60, flush=True)
    print(f"Total latency:  {elapsed:.1f} s", flush=True)
    print(f"Token chunks:   {len(tokens)}", flush=True)
    print(f"Generation len: {len(text)} chars", flush=True)
    print(flush=True)
    print("First 400 chars of answer:", flush=True)
    print(text[:400], flush=True)

    if final:
        state = final["state"]
        print(flush=True)
        print(f"Docs retrieved: {len(state.get('documents', []))}", flush=True)
        print(f"Confidence:     {state.get('confidence_score', 0.0):.2f}", flush=True)
        print(f"Citations:      {len(state.get('citations', []))}", flush=True)
        print(f"Needs review:   {state.get('needs_human_review')}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
