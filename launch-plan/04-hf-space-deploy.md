# 04 — Phase 3: HF Space Dockerfile + Push

**Owner of this phase:** AI agent.
**Pre-requisite:** phase 2 (backend BYOK mode) complete and tests green.

## Goal

Build a Docker image that runs the FastAPI backend on HF Space port 7860 with all production dependencies (no Ollama, no Postgres, no Redis). Push to `huggingface.co/spaces/LeomordKaly/secureagentrag-api`. Confirm reachable from Egypt.

## Differences from existing `Dockerfile`

The current `Dockerfile` was written for local docker-compose and assumes:

- Streamlit on port 8501
- Ollama at `http://ollama:11434` inside the compose network
- Qdrant at `http://qdrant:6333` inside the compose network
- Postgres + Redis inside the compose network

For HF Space we need:

- No Streamlit — FastAPI only on port 7860
- No Ollama — embeddings via local `sentence-transformers` BGE-M3, LLM calls go out to Groq via Cloud
- Qdrant via Qdrant Cloud (HTTPS, API key) — no local container
- No Postgres, no Redis — SQLite on `/tmp` (HF Space writeable area)
- No GPU — all models on CPU
- No Phoenix — `SAR_BYOK_MODE=true` disables it

## File: `Dockerfile.hf`

```dockerfile
# ===============================================================
# Stage 1: builder
# ===============================================================
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
COPY uv.lock ./

# Install only the optional groups we need for prod:
# - api (FastAPI, uvicorn)
# - embeddings-local (sentence-transformers for BGE-M3)
# - pii (presidio for redaction)
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        -e ".[api,embeddings-local,pii]"

# ===============================================================
# Stage 2: runtime
# ===============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# HF Spaces require user 1000:1000 for writeable directories
RUN useradd -m -u 1000 user

# System deps for PDF / OCR fallback
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1-mesa-glx curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Pre-download the reranker checkpoint so the first request does not pay
# the ~2 GB download tax. The fine-tuned reranker is NOT checked into git
# (gitignored); pull it from a public HF Hub repo we own.
# Path inside the container: /app/data/checkpoints/reranker-domain-v1
RUN mkdir -p /app/data/checkpoints && \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download(repo_id='LeomordKaly/secureagentrag-reranker-v1', \
                                 local_dir='/app/data/checkpoints/reranker-domain-v1')"

COPY --chown=user:user . .

USER user

ENV SAR_BYOK_MODE=true
ENV SAR_QDRANT_URL="https://placeholder.cloud.qdrant.io"
ENV SAR_QDRANT_API_KEY="placeholder"
ENV SAR_RERANKER_TYPE=fine_tuned
ENV SAR_FINETUNED_RERANKER_PATH=/app/data/checkpoints/reranker-domain-v1
ENV SAR_LLM_MODEL=llama-3.1-8b-instant
ENV SAR_CLOUD_PROVIDER=groq
ENV SAR_DEFAULT_PROVIDER=groq
ENV SAR_AUDIT_LOG_DIR=/tmp/audit_logs
ENV SAR_CONVERSATION_DIR=/tmp/conversations
ENV SAR_CHECKPOINT_DB_PATH=/tmp/checkpoints.sqlite
ENV SAR_CORS_ALLOW_ORIGINS='["https://secureagentrag.vercel.app"]'

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "interfaces.api:app", "--host", "0.0.0.0", "--port", "7860"]
```

## File: `README.md` for the HF Space

HF Spaces require a YAML front-matter in `README.md`:

```markdown
---
title: SecureAgentRAG API
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Privacy-first multi-agent RAG backend with RBAC, faithfulness gate, and audit chain
---

# SecureAgentRAG API

FastAPI backend for the [secureagentrag](https://github.com/moazmo/secureagentrag) demo.

Frontend: https://secureagentrag.vercel.app
Source code: https://github.com/moazmo/secureagentrag
```

This file lives at the **HF Space repo root**, not in the GitHub repo.

## Setting Secrets in HF Space

The HF Space settings panel lets you define secrets that are exposed as env vars at runtime. Set:

| Secret name | Source |
|---|---|
| `SAR_QDRANT_URL` | Phase 1 smoke `SAR_QDRANT_CLOUD_URL` value |
| `SAR_QDRANT_API_KEY` | Phase 1 smoke `SAR_QDRANT_CLOUD_API_KEY` value |
| `SAR_GROQ_API_KEY` | Existing key from local `.env` |

These override the placeholders in the Dockerfile.

## Pre-deploy: push the reranker checkpoint to HF Hub

The fine-tuned reranker is 2.27 GB. It is gitignored locally and absent from the GitHub repo. To make the Space build reproducible, push the checkpoint to a separate HF Hub model repo we own:

```bash
# One-time setup, owner runs locally
huggingface-cli login                          # uses HF_TOKEN
huggingface-cli upload LeomordKaly/secureagentrag-reranker-v1 \
    data/checkpoints/reranker-domain-v1 . \
    --repo-type model
```

Make the model repo **public** so the HF Space build can download it without auth.

## Build and push the Space

The Space repo is a separate git remote at `https://huggingface.co/spaces/LeomordKaly/secureagentrag-api`.

We use a deploy script (`scripts/deploy_hf_space.py`) that:

1. Builds a clean sparse worktree containing only the files the Space needs:
   - `interfaces/`, `inference/`, `retrieval/`, `core/`, `config/`, `utils/`, `evaluation/calibration.json`
   - `Dockerfile.hf` renamed to `Dockerfile`
   - `pyproject.toml`, `uv.lock`
   - HF-flavored `README.md`
2. Force-pushes to the Space remote (`huggingface.co/spaces/LeomordKaly/secureagentrag-api`)
3. Polls the Space build status until "Running" or "Error"
4. On success, prints the live URL

```bash
uv run python scripts/deploy_hf_space.py --check-only         # dry run, no push
uv run python scripts/deploy_hf_space.py                       # push
```

## Acceptance criteria

- [ ] `Dockerfile.hf` builds locally with `docker build -f Dockerfile.hf -t sar-hf .`
- [ ] Local container responds: `docker run -p 7860:7860 -e SAR_QDRANT_URL=... -e SAR_QDRANT_API_KEY=... sar-hf` → `curl localhost:7860/health` returns 200
- [ ] HF Space build succeeds (check at `https://huggingface.co/spaces/LeomordKaly/secureagentrag-api/logs`)
- [ ] `curl https://LeomordKaly-secureagentrag-api.hf.space/health` from Egypt returns 200
- [ ] BYOK smoke from Egypt: `curl -H "X-User-LLM-Key: <test-key>" -H "X-User-Provider: groq" ...` returns a streamed response
- [ ] HF Space settings → secrets contain `SAR_QDRANT_URL`, `SAR_QDRANT_API_KEY`, `SAR_GROQ_API_KEY`

## Cold-start mitigation

The HF Space sleeps after 48 hours of zero traffic. Phase 6 sets up a GitHub Actions cron pinging `/health` every 24 hours. See `07-keepalive-cron.md`.

## Image size budget

CPU Basic Spaces have 50 GB ephemeral disk. Our image budget:

- Python 3.11-slim base: ~150 MB
- BGE-M3 embedding model weights: ~1.5 GB (downloaded on first import, cached to `/home/user/.cache/huggingface`)
- Fine-tuned reranker: ~2.3 GB (downloaded at build time per Dockerfile)
- FastAPI + langgraph + qdrant-client + sentence-transformers + dependencies: ~3 GB
- **Total runtime memory:** ~7 GB. Image-on-disk: ~6 GB. Well under HF limits.

## Time-on-first-request budget

| Step | Cold (post-wake) | Warm |
|---|---|---|
| Embedding model load | ~5 s | 0 |
| Reranker load | ~2 s | 0 |
| Qdrant search | ~50 ms | ~50 ms |
| LLM call (Groq) | ~500 ms | ~500 ms |
| Faithfulness NLI | ~200 ms | ~200 ms |
| **Total first request** | **~8 s** | **~1 s** |

Cold start of ~8 s is acceptable when sleep boundary is 48 h. With keepalive cron in place, visitors hit warm path 99% of the time.
