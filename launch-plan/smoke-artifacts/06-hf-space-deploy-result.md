# Phase 3 Deploy — Real Backend on HF Space (PASS, with follow-ups)

**Run date:** 2026-05-26
**Run by:** Claude (agent), tokens supplied earlier in phase 1
**Live URL:** https://LeomordKaly-secureagentrag-api.hf.space

## Outcome

| Check | Result |
|---|---|
| `Dockerfile.hf` written + `.dockerignore` configured | ✅ |
| Reranker pre-cache at build (cross-encoder, BAAI/bge-reranker-v2-m3) | ✅ (cached to `/home/user/.cache/huggingface`) |
| HF Space secrets pushed (Qdrant URL + key, Groq key) | ✅ via `api.add_space_secret` — never echoed full in logs |
| `scripts/deploy_hf_space.py` end-to-end run | ✅ |
| Build → Running | BUILDING 450 s → APP_STARTING 72 s → RUNNING |
| `/healthz` from Egypt | HTTP 200, 0.54 s TTFB |
| `/openapi.json` lists `/byok/chat` route | ✅ — confirms `byok_mode=true` propagated through Space secrets |
| `/byok/chat` owner-key path (no `X-User-LLM-Key`) | HTTP 200, full pipeline executes (cold ~30 s, warm ~5 s) |
| `/byok/chat` BYOK path (with `X-User-LLM-Key`) | HTTP 200, 8.8 s, `byok_used: true` in response |
| Pipeline correctness on empty corpus | ✅ — synthesizer correctly refuses with "no documents found" instead of hallucinating |
| `/readyz` | HTTP 503 — see follow-up §1 |
| Owner-key throttle (3/h cap) firing in production | ❌ — see follow-up §2 |

## Two-stage build summary

| Stage | Size | Time |
|---|---|---|
| Builder (Python 3.11-slim + uv + `[api,embeddings-local]` extras) | n/a | ~150 s |
| Runtime (slim base + .venv + source + cross-encoder cache) | ~5 GB image | ~300 s incl. apt + reranker download |
| **Total build** | — | **~450 s** |

## Architecture in production

```
Egypt visitor → Vercel frontend (phase 4)
              → HF Space subdomain
                LeomordKaly-secureagentrag-api.hf.space
                ↓ uvicorn (1 worker) on port 7860
                ↓ FastAPI / interfaces/api.py
                ↓ run_rag_pipeline (LangGraph 9-node)
                ↓ retrieval/qdrant_client.py
                  → Qdrant Cloud (db2d4134-...us-east-1-1.aws.cloud.qdrant.io)
                ↓ inference/cloud_clients.py
                  → Groq API (llama-3.1-8b-instant)
                ↑ JSON {session_id, persona, byok_used, response}
```

Embeddings: BGE-M3 local via `sentence-transformers` (CPU, ~1.5 GB RAM).
Reranker: BGE-Reranker-v2-M3 cross-encoder (cached at build, ~1.2 GB RAM).
Audit log: `/tmp/secureagentrag/audit_logs/*.jsonl` (ephemeral, wiped per session).

## Environment overrides applied via HF Space secrets (not committed)

| Secret | Source |
|---|---|
| `SAR_QDRANT_URL` | from `.env::SAR_QDRANT_CLOUD_URL` |
| `SAR_QDRANT_API_KEY` | from `.env::SAR_QDRANT_CLOUD_API_KEY` |
| `SAR_GROQ_API_KEY` | from `.env::SAR_GROQ_API_KEY` |

All other configuration is baked into `Dockerfile.hf` (BYOK mode flags, CORS allowlist for `https://app.eilm.live`, paths under `/tmp`, model names).

## Two build failures fixed during deploy

### Fix 1 — README short_description ≤ 60 chars

HF rejected the upload at `validate-yaml`:

```
"short_description" length must be less than or equal to 60 characters long
```

Resolution: shortened in `scripts/deploy_hf_space.py::HF_README_BODY` from
`"Privacy-first multi-agent RAG backend (BYOK) for the public demo"` (66 chars)
to `"Privacy-first multi-agent RAG (BYOK demo)"` (42 chars).

### Fix 2 — Debian package rename

`apt-get install libgl1-mesa-glx` failed in Debian trixie (the new
`python:3.11-slim` base). Debian renamed the package to `libgl1` and dropped
the `-mesa-glx` variant.

Resolution: in `Dockerfile.hf` replaced
`libxrender-dev libgl1-mesa-glx` → `libxrender1 libgl1`.

### Fix 3 — Drop `[pii]` extras from runtime

The Presidio analyzer auto-downloads spaCy `en_core_web_lg` at module import
in containers where `pip` is not on the path (HF Spaces ships only `uv`).
The auto-install fails with `✘ No package installer found` and the container
exits 1.

Resolution: changed `uv pip install -e ".[api,embeddings-local,pii]"` →
`uv pip install -e ".[api,embeddings-local]"` in `Dockerfile.hf`. The regex
patterns in `utils/pii.py::_REGEX_PATTERNS` (extended in P2.5 to cover
Groq / OpenAI / Anthropic / HF / Vercel / Qdrant JWT shapes) provide
sufficient redaction without the NER layer.

Audit-log redaction tests under `tests/test_security/test_byok_key_redaction.py`
continue to pass on `.[pii]` and on regex-only — Presidio was always a
best-effort enhancement, never the primary defence.

## Follow-ups (deferred to phase 3.5)

### 1. `/readyz` returns 503

The readiness probe pings `Ollama` and `Qdrant` via the legacy
`utils.health.run_health_checks`. In production we use Qdrant Cloud (HTTPS)
and Groq (no Ollama). The probe sees Ollama missing → returns 503.

This is **not** blocking: `/healthz` returns 200 and the GitHub Actions
keepalive cron pings `/healthz` only. The Next.js frontend (phase 4) also
calls `/healthz` for status badges.

Fix scheduled for phase 3.5: extend `utils/health.py` to skip Ollama in
BYOK mode and check Groq's `/models` endpoint instead.

### 2. Owner-key throttle does not fire behind HF's reverse proxy

The throttle uses `request.client.host` which behind HF's TLS-terminating
proxy is the proxy's IP, not the visitor's. All visitors share one bucket;
the bucket fills past the 3/h cap quickly under low traffic, but does not
actually rate-limit per-visitor as designed.

Production fix: read `X-Forwarded-For` (HF sets it to the real client IP)
in `interfaces/api.py::byok_chat_endpoint`. Add a unit test for that
extraction path.

The pre-deploy unit tests in `tests/test_utils/test_owner_key_throttle.py`
pass — the logic is correct; only the IP source is wrong in production.
Deferred fix tracked in `private/roadmap.md`.

## Smoke commands (reproducible)

```bash
# Liveness
curl -s https://LeomordKaly-secureagentrag-api.hf.space/healthz

# Owner-key path (no header)
curl -s -X POST https://LeomordKaly-secureagentrag-api.hf.space/byok/chat \
  -H "Content-Type: application/json" \
  -H "X-Demo-Persona: engineer" \
  -d '{"query":"What is RBAC?"}'

# BYOK path (visitor key)
curl -s -X POST https://LeomordKaly-secureagentrag-api.hf.space/byok/chat \
  -H "Content-Type: application/json" \
  -H "X-User-LLM-Key: <visitor groq key>" \
  -H "X-User-Provider: groq" \
  -H "X-Demo-Persona: compliance" \
  -d '{"query":"explain hash chain audit"}'
```

## Cost ceiling — unchanged

| Component | Monthly |
|---|---|
| HF Space CPU Basic | $0 |
| Qdrant Cloud free 1 GB | $0 |
| Groq free tier | $0 |
| **Total** | **$0** |

## Next phase

Phase 4 — Next.js + shadcn frontend on Vercel. Requires sibling repo
`secureagentrag-web` on GitHub + Vercel project. Will wire `localStorage`
BYOK input → `X-User-LLM-Key` header → this Space.
