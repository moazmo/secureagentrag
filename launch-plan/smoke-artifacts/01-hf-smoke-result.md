# Phase 1 Smoke — Hugging Face Spaces (PASS)

**Run date:** 2026-05-26
**Run by:** Claude (agent), token supplied by owner

## Outcome

| Check | Result |
|---|---|
| Account signup (no CC required) | ✅ `LeomordKaly` (email `moazmo27@gmail.com`), `canPay: False`, `plan: None` — confirms free tier with no card on file |
| Write-scope access token | ✅ `hf_pyrM…tdiO` (truncated for log) |
| Space creation via `HfApi.create_repo` | ✅ `https://huggingface.co/spaces/LeomordKaly/secureagentrag-api` |
| Docker SDK + CPU Basic auto-assigned | ✅ `hardware=cpu-basic` (2 vCPU, 16 GB RAM) |
| Build → start time | ~24 seconds total (BUILDING → APP_STARTING → RUNNING) |
| Public URL reachable from Egypt | ✅ `https://LeomordKaly-secureagentrag-api.hf.space/health` returned HTTP 200 |
| Latency from Egypt origin | TLS handshake 0.41 s, TTFB 0.55 s, total 0.55 s — well within budget |

## Artifacts uploaded

| File | Size | Purpose |
|---|---|---|
| `Dockerfile` | 208 B | Python 3.11-slim + FastAPI + uvicorn, EXPOSE 7860 |
| `main.py` | 1039 B | `GET /` + `GET /health` returning JSON with uptime |
| `README.md` | 758 B | YAML frontmatter declares `sdk: docker`, `app_port: 7860` |

Files live at `launch-plan/smoke-artifacts/hf-hello/` and are tracked in this branch so the agent and the next AI agent both see what shipped.

Upload commit on Space repo: `17d9fadbe279aaa2e69b4da7c72b2d7f9655f16f`

## Sample response from `/health`

```json
{
  "status": "ok",
  "service": "secureagentrag-api",
  "phase": "1-smoke",
  "python": "3.11.15",
  "uptime_seconds": 14.14
}
```

## What this proves

1. **No credit card required on HF free tier in 2026.** `canPay: False` plus `plan: None` confirms the account was created without payment info. Signal still green.
2. **CPU Basic hardware is the default.** The Space picked `cpu-basic` automatically; no manual hardware selection needed.
3. **Docker SDK works end-to-end via `HfApi.upload_folder`.** No git clone needed — `HfApi.create_repo` + `upload_folder` ship code straight from the local working tree.
4. **Cold build is ~24 s.** Faster than the ~60 s budgeted in `04-hf-space-deploy.md`. Real backend image will be larger (~6 GB) and likely take 2-5 min on first build, but that is one-time on push, not on every wake.
5. **Egypt-to-HF round-trip is ~550 ms.** Acceptable for a chat UI that streams tokens after ~500 ms TTFB on the LLM call.

## Important note on username

The plan in `launch-plan/` was drafted assuming HF username `moazmo` to match the GitHub identity. The actual HF account is `LeomordKaly`. All HF-related URLs in the plan need updating from `moazmo-secureagentrag-api.hf.space` to `LeomordKaly-secureagentrag-api.hf.space`. This is a doc-only change — no code impact — and is being applied in the same commit that lands this smoke result.

GitHub identity remains `moazmo` (`moazmo27@gmail.com`). HF identity is `LeomordKaly` (same email).

## Next phase

Phase 1 smoke #2: Qdrant Cloud signup + 1 GB cluster + sparse-vector smoke. Needs owner to sign up at `cloud.qdrant.io` and paste back the cluster URL + API key.

## Cleanup considerations (post-launch only)

The smoke Dockerfile + hello-world will be overwritten by the Phase 3 real backend image. No need to delete the Space — same repo gets the real code on the next `upload_folder` call.
