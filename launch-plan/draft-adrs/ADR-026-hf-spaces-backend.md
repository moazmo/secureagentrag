# ADR-026 (DRAFT): Hugging Face Spaces as Production Backend Host

**Date:** 2026-05-25 (drafted) / TBD (accepted)
**Status:** **Draft** — flips to **Accepted** when the HF Space serves `/health` returning 200 from Egypt.

## Context

The production launch requires a backend host that meets four hard constraints:

1. **Zero USD** — no monthly, no annual, no metered surprise bills
2. **No credit card at signup** — eliminates AWS, GCP, Azure, Oracle Cloud, Fly.io, Railway, Koyeb (in many regions including Egypt)
3. **Available in Egypt** — eliminates anything with regional signup gating
4. **No Render-style cold start** — Render sleeps after 15 minutes inactivity and takes 30-60s to wake, which is unacceptable for a recruiter-facing demo

A free-tier audit run on 2026-05-25 found that the surviving options were:

| Service | Notes |
|---|---|
| Hugging Face Spaces (Docker, CPU Basic) | 16 GB RAM, 48-hour idle sleep, no CC, Egypt OK |
| Northflank Sandbox | claimed always-on, CC signup unclear |
| Streamlit Community Cloud | shorter sleep than HF, Streamlit-only |
| Vercel Hobby (Python serverless) | 60-second function timeout — too short for our RAG pipeline |
| Cloudflare Pages | Python via Pyodide only, won't fit our dependency tree |

## Decision

Use **Hugging Face Spaces with the Docker SDK on CPU Basic hardware** for the FastAPI backend.

- Space URL: `huggingface.co/spaces/moazmo/secureagentrag-api`
- Public subdomain: `moazmo-secureagentrag-api.hf.space`
- Hardware: CPU Basic (2 vCPU, 16 GB RAM, $0/mo)
- SDK: Docker
- Container port: 7860 (HF default)
- Dockerfile: `Dockerfile.hf` in the GitHub repo
- Sleep mitigation: GitHub Actions cron pings `/health` every 24 hours
- Secrets: `SAR_QDRANT_URL`, `SAR_QDRANT_API_KEY`, `SAR_GROQ_API_KEY` set via HF Space settings panel
- Reranker checkpoint hosted on a separate **public** HF Hub model repo (`moazmo/secureagentrag-reranker-v1`) and downloaded at Docker build time

## Consequences

- (+) Zero recurring cost
- (+) 16 GB RAM fits BGE-M3 + fine-tuned reranker + FastAPI + Python stack with ~10 GB headroom
- (+) `git push` deploy — same Dockerfile workflow we already use
- (+) Free `.hf.space` subdomain — no domain purchase required for portfolio
- (+) HF Spaces is portfolio-credible (used by major ML demos)
- (+) Sleep boundary is 48 hours, not 15 minutes — 192× more forgiving than Render
- (-) 30-60 second cold-start on first wake. Mitigated by GitHub Actions keepalive cron
- (-) Ephemeral disk — audit log + checkpoints wiped per restart. Acceptable for BYOK demo, intentional alignment with privacy story
- (-) No GPU on free tier. Fine-tuned reranker runs ~5× slower on CPU but still <500ms at top_k=10
- (-) Custom domain requires HF Pro ($9/mo). Acceptable — we use `.hf.space` + a separate `.vercel.app` for the frontend
- (-) Tied to HF's roadmap. If HF degrades the CPU Basic tier, we have to migrate. Fallback: Northflank Sandbox

## Alternatives considered

- **Oracle Cloud Always Free** — required credit card at signup. Rejected by owner constraint
- **Render free tier** — 15-minute sleep + 30-60s wake makes the demo unusable for cold recruiter traffic
- **Koyeb Hobby** — CC required in many regions including Egypt
- **Northflank Sandbox** — promising but CC requirement unclear; held in reserve as fallback
- **Self-host on Hostinger Business shared** — no Docker, no FastAPI, no long-running Python; only suitable for the static landing page

## Acceptance criteria for flip to Accepted

- HF Space provisioned and reachable from Egypt
- `Dockerfile.hf` builds cleanly
- `curl https://moazmo-secureagentrag-api.hf.space/health` returns 200 from Egypt
- Phase 2 BYOK backend runs successfully on the Space
- ADR-026 moved from `launch-plan/draft-adrs/` into `DECISIONS.md` with this content

## Related ADRs

- ADR-025 (BYOK demo mode) — the ephemeral-disk + 48h-sleep constraints shape BYOK's session model
- ADR-027 (Vercel + Next.js frontend) — separated frontend host since HF Spaces is backend-only here
- ADR-028 (Qdrant Cloud + session collections) — Qdrant Cloud chosen because HF Space disk is ephemeral
