"""Warm up Ollama by pinning the LLM and embedding models into VRAM.

Run this once after bringing Ollama up — it eliminates the 5-10 second
cold-load penalty on the first Streamlit query.

Usage:
    uv run python -m scripts.warmup_ollama
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.settings import settings  # noqa: E402


def main() -> int:
    base = settings.ollama_url.rstrip("/")
    keep_alive = settings.ollama_keep_alive
    print(f"Ollama: {base}")
    print(f"keep_alive: {keep_alive}")
    print(f"LLM:       {settings.llm_model}")
    print(f"Embedding: {settings.embedding_model}")
    print("-" * 60)

    with httpx.Client(timeout=180.0) as client:
        t = time.time()
        print(f"Warming embedding ({settings.embedding_model})... ", end="", flush=True)
        r = client.post(
            f"{base}/api/embed",
            json={"model": settings.embedding_model, "input": "warmup", "keep_alive": keep_alive},
        )
        r.raise_for_status()
        print(f"OK ({time.time() - t:.1f}s)")

        t = time.time()
        print(f"Warming LLM ({settings.llm_model})... ", end="", flush=True)
        r = client.post(
            f"{base}/api/generate",
            json={
                "model": settings.llm_model,
                "prompt": "hello",
                "stream": False,
                "keep_alive": keep_alive,
                "options": {"num_predict": 8},
            },
        )
        r.raise_for_status()
        print(f"OK ({time.time() - t:.1f}s)")

        print("-" * 60)
        print("Currently loaded:")
        r = client.get(f"{base}/api/ps")
        for m in r.json().get("models", []):
            vram_mb = m.get("size_vram", 0) / 1024 / 1024
            print(f"  {m.get('name'):25s} vram={vram_mb:6.0f} MB  expires_at={m.get('expires_at')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
