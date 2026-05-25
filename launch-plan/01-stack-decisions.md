# 01 — Stack Decisions

## Constraints (hard, in priority order)

1. **Zero USD** — no monthly, no annual, no metered surprise bills
2. **No credit card at signup** — eliminates Oracle, Koyeb (region-dependent), AWS, GCP, Azure, Fly.io, Railway
3. **Available in Egypt** — eliminates any provider with regional gating at signup or runtime
4. **No Render-style cold start** — Render sleeps after 15 minutes inactivity and takes 30-60 seconds to wake. Recruiter visiting cold = bad first impression
5. **Hostinger Business Web Hosting** — owner already paid annually; should be used if possible

## Free-tier matrix (verified 2026-05-25)

| Service | No CC? | Cold start | Python | Egypt | Verdict |
|---|---|---|---|---|---|
| Hugging Face Spaces (Docker, CPU Basic) | yes | 48-hour idle sleep, ~30-60s wake (mitigated by keepalive cron) | yes (Docker) | yes | **SELECTED — backend** |
| Qdrant Cloud free tier | yes | always-on | n/a (vector DB) | yes | **SELECTED — vector store** |
| Groq free tier | yes (key already in `.env`) | always-on, 30 RPM | n/a (LLM API) | yes | **SELECTED — owner-key LLM** |
| Vercel Hobby | yes | none (Edge/SSR), ~500ms (serverless) | limited (60s timeout, no long-running) | yes | **SELECTED — frontend** |
| Hostinger Business Web Hosting | already paid | none | shared LiteSpeed, no FastAPI, no Docker | yes | **SELECTED — static landing only** |
| Streamlit Community Cloud | probable yes | apps sleep after inactivity | Streamlit-only | yes | rejected (worse sleep than HF) |
| Cloudflare Pages | yes | none | Pyodide beta only, won't fit our deps | yes | rejected (Python support too thin) |
| Northflank Sandbox | unclear at signup | claimed always-on | yes (Docker) | yes | **BACKUP** if HF Space fails smoke |
| Koyeb Hobby | region-dependent, Egypt likely on CC list | scale-to-zero | yes | unclear | rejected (CC risk) |
| Oracle Cloud Always Free | no (CC required) | none | yes | yes | rejected (user excluded) |
| Render Free | yes | 15-minute sleep, 30-60s wake | yes | yes | rejected (user excluded) |
| Fly.io | no (CC required since 2024) | none | yes | yes | rejected |
| Railway | no (CC required since 2023) | none | yes | yes | rejected |
| AWS / GCP / Azure free tier | no (CC required) | varies | yes | yes | rejected |
| PythonAnywhere free | yes | none for web apps | yes but WSGI-only, no websockets | yes | rejected (no FastAPI streaming) |
| Heroku | no (paid since 2022) | n/a | yes | yes | rejected |
| Glitch | yes | 5-minute sleep | yes | yes | rejected (sleep worse than Render) |

## Final stack

```
┌──────────────────────────────────────────────────────────────────┐
│ Hostinger Business Web Hosting                                   │
│ static landing page: hero + screenshots + YouTube + CTA          │
│ DNS Zone Editor available for future custom domain               │
└──────────────────────────────────────────────────────────────────┘
                       │ <a href> CTA "Open live demo →"
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Vercel Hobby                                                     │
│ Next.js 15 App Router + shadcn/ui + Tailwind + Vercel AI SDK     │
│ - localStorage: BYOK key + provider choice + Ollama URL          │
│ - useChat() with streaming SSE                                   │
│ - persona switcher, audit-chain viewer, mobile responsive        │
│ - URL: secureagentrag-<owner>.vercel.app                         │
└──────────────────────────────────────────────────────────────────┘
                       │ HTTPS POST /chat/stream
                       │ X-User-LLM-Key: <visitor BYOK or owner-fallback>
                       │ X-User-Provider: groq|openai|anthropic|ollama
                       │ X-Session-ID: <client-generated UUID>
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Hugging Face Space (Docker SDK, CPU Basic, 2 vCPU / 16 GB)       │
│ FastAPI on port 7860                                             │
│ - SAR_BYOK_MODE=true                                             │
│ - per-IP throttle: 3 owner-key queries/hour, unlimited w/ BYOK   │
│ - session middleware → documents_<session_id> Qdrant collection  │
│ - 24h purge cron (HF Scheduled Spaces task)                      │
│ - bge-m3 embeddings local in RAM                                 │
│ - fine-tuned reranker on CPU (~5× slower than GPU, still <500ms) │
│ - SQLite audit + checkpoint, wiped per session                   │
│ - Phoenix observability disabled in BYOK mode                    │
│ - URL: huggingface.co/spaces/moazmo/secureagentrag-api           │
│ - Wrapped: moazmo-secureagentrag-api.hf.space                    │
└──────────────────────────────────────────────────────────────────┘
                       │ qdrant-client (HTTPS, API key)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Qdrant Cloud free tier (1 GB / 1M vectors)                       │
│ - dense + sparse hybrid (same SDK, same schema)                  │
│ - RBAC payload filter unchanged                                  │
│ - per-session collection cleanup via DELETE after 24h            │
└──────────────────────────────────────────────────────────────────┘
                       │ Groq API client (OpenAI-compatible)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│ Groq free tier (30 RPM org-wide / 14,400 req/day)                │
│ Owner key in .env, visitors can BYOK to skip throttle            │
│ Also accepts visitor OpenAI / Anthropic / Ollama-URL keys        │
└──────────────────────────────────────────────────────────────────┘
```

## Cost ceiling

| Line | Monthly cost |
|---|---|
| HF Space CPU Basic | $0 |
| Qdrant Cloud free tier | $0 |
| Vercel Hobby plan | $0 |
| Groq free tier | $0 |
| Hostinger Business plan | already paid (~$5/mo amortized over yearly plan) |
| GitHub Actions keepalive | $0 (≤ 50 min/mo against 2000 min/mo free quota) |
| Custom domain | $0 (use `.hf.space` + `.vercel.app`) |
| **New spend** | **$0** |

## Why HF Spaces wins single-handedly

1. **Confirmed no-CC signup** by multiple sources and Hugging Face's own pricing page
2. **16 GB RAM, 2 vCPU CPU Basic free** — fits BGE-M3 embeddings (~1.5 GB) + fine-tuned reranker (~600 MB on CPU) + FastAPI + Python stack with ~12 GB headroom
3. **Egypt-accessible** — HF Hub is globally distributed, no regional gate
4. **48-hour idle sleep, not 15-minute** — 192× more forgiving than Render. A free 5-line GitHub Actions cron pinging every 24 hours eliminates sleep entirely
5. **`git push` deploy** — existing `Dockerfile` is 90% compatible, only port + entrypoint change needed
6. **Free `.hf.space` subdomain** — no domain purchase required for portfolio
7. **Docker SDK** — full control over the runtime, can install any pip package, can run any process, no Passenger / WSGI restrictions

## Why Vercel for frontend specifically

1. **Confirmed no-CC Hobby plan** in 2026
2. **Zero cold start** for Next.js SSR and static pages
3. **Vercel AI SDK** ships SSE streaming, persona toggles, useChat hook out of the box — saves a week of frontend plumbing
4. **`vercel.app` subdomain free** — no domain needed
5. **Industry-standard portfolio signal** — recruiters expect "Next.js on Vercel" for AI products in 2026

## Why we explicitly do NOT use Hostinger Business for compute

Hostinger Business Web Hosting is a **shared LiteSpeed plan**, NOT VPS. Confirmed limitations:

- No long-running Python processes (Passenger WSGI only — kills idle workers)
- No FastAPI / ASGI support (websockets blocked on shared)
- No Docker
- No GPU
- No SSH (only SFTP)
- Streamlit specifically known broken on Hostinger shared

What it **can** do for this project:

- Static HTML hosting (the landing page)
- DNS Zone Editor (CNAME records for free subdomains)
- File Manager via hPanel (upload static files)
- Free SSL on owner's domain (if any)

## Stop conditions and fallbacks

| Stop condition | Fallback |
|---|---|
| HF Space signup asks for CC | Try Northflank Sandbox (Docker, no sleep per docs) |
| HF Space drops 16 GB CPU Basic | Switch to Northflank Sandbox or shrink Python stack (drop sentence-transformers, use Groq embeddings) |
| Qdrant Cloud signup asks for CC | Self-host Qdrant in same HF Space (data lost on restart — acceptable for stateless demo) |
| Qdrant Cloud drops sparse vector support on free | Use Qdrant binary self-hosted in HF Space |
| Groq adds CC requirement | Visitors must BYOK only; no owner-key fallback |
| Vercel adds CC requirement | Deploy Next.js as static export to Hostinger Business |
| Egypt blocked by any provider at runtime | Use Cloudflare Tunnel from owner's local network as backup origin |

## Architecture decision records to write (post-implementation)

These will be added to `DECISIONS.md` once each phase ships:

- **ADR-025: BYOK demo mode** — rationale, per-request key extraction, session-scoped Qdrant collections, audit redaction tests
- **ADR-026: Hugging Face Spaces as backend host** — why this beat Oracle Free + Render + Koyeb + Northflank for the no-CC + no-cold-start + Egypt-OK matrix
- **ADR-027: Frontend split to Vercel + Next.js** — why we dropped Streamlit for the public demo
- **ADR-028: Qdrant Cloud free tier + session collections** — why this beat self-hosted Qdrant in the HF Space

Until shipped, these are **draft ADRs** living in this folder (`launch-plan/draft-adrs/`) so the main `DECISIONS.md` does not claim work that has not happened.
