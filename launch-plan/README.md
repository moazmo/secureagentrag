# SecureAgentRAG — Production Launch Plan

> **Branch:** `deploy/prod-launch` (this branch). `main` frozen at `56c8c98` (last-known-good Streamlit-only state).
>
> **Status:** plan accepted 2026-05-25 — pre-implementation. Smoke tests pending.
>
> **Owner:** Moaz Muhammad (`moazmo27@gmail.com`, GitHub `moazmo`).
>
> **Constraints (hard):** zero USD, no credit card at signup, available in Egypt, no Render-style 15-min cold start.

This directory is the single source of truth for the production launch. **Any AI agent picking up this work — Claude Code, Kilo CLI, Hermes, Antigravity, GPT, Gemini — must read every file in this folder in numeric order before touching any other code.**

## Quick context

SecureAgentRAG is a privacy-first multi-agent RAG platform at ~29.5k Python LOC across 147 files, 487 tests, 24 ADRs, 9-node LangGraph state machine. P1-P5 of the original roadmap shipped on `main`. This launch is P6 — the production deployment phase.

The goal of P6: replace the local Streamlit demo with a public Next.js BYOK frontend on Vercel talking to a FastAPI backend on Hugging Face Spaces, using Qdrant Cloud for the vector store and Groq for owner-key fallback LLM access. Recruiters land on a static Hostinger landing page that links into the live demo.

## Files in this folder (read in order)

| File | What it covers |
|---|---|
| `00-context.md` | Current repo state, what is and is not changing |
| `01-stack-decisions.md` | Full free-tier audit and final stack rationale |
| `02-smoke-tests.md` | Five signup + smoke deploys to run **before any code** |
| `03-backend-byok.md` | FastAPI BYOK mode spec — per-request key extraction, session collections, audit redaction |
| `04-hf-space-deploy.md` | `Dockerfile.hf` + HF Space `huggingface.co/spaces/moazmo/secureagentrag-api` setup |
| `05-nextjs-frontend.md` | Next.js 15 + shadcn + Vercel AI SDK frontend spec |
| `06-hostinger-landing.md` | Static landing page spec + Hostinger hPanel deploy steps |
| `07-keepalive-cron.md` | GitHub Actions workflow to keep HF Space warm |
| `08-demo-video.md` | 4-minute screencast brief — record against live deploy |
| `09-doc-sweep.md` | List of every `.md` file to update after launch (public + private) |
| `10-rollback-plan.md` | How to revert if anything in production breaks |
| `11-security-checklist.md` | BYOK key safety, CORS, audit redaction, rate limiting |
| `12-agent-handoff.md` | Explicit instructions for any AI agent continuing this work |

## Accepted decisions (locked 2026-05-25)

1. **Frontend:** rebuild as Next.js 15 + shadcn/ui on Vercel free tier. Not Streamlit.
2. **LLM access for visitors:** both. Owner-key throttled at 3 queries per IP per hour, plus unlimited BYOK unlock when visitor pastes their own key.
3. **Landing:** static HTML on existing Hostinger Business Web Hosting plan, linking to the Vercel demo URL.
4. **Demo video:** record against the live deploy, not against the local Streamlit.
5. **Branch:** all work on `deploy/prod-launch`. `main` stays frozen at `56c8c98` until live smoke passes.

## Build phases

```
Phase 0  — Plan written, doc sweep (this commit)
Phase 1  — Smoke signups (HF, Qdrant Cloud, Vercel, Groq verify, Hostinger panel)  ← USER ACTION
Phase 2  — Backend BYOK mode (deploy/prod-launch)                                  ← AGENT
Phase 3  — Dockerfile.hf + HF Space push                                            ← AGENT
Phase 4  — Next.js BYOK frontend + Vercel deploy                                    ← AGENT
Phase 5  — Hostinger static landing page                                            ← AGENT (HTML), USER (upload via hPanel)
Phase 6  — GitHub Actions keepalive cron                                            ← AGENT
Phase 7  — End-to-end smoke from Egypt                                              ← BOTH
Phase 8  — Demo video                                                                ← USER (record), AGENT (edit script)
Phase 9  — Final doc sweep + merge to main                                          ← AGENT
```

## Cost ceiling

| Line | Monthly |
|---|---|
| HF Space CPU Basic (2 vCPU / 16 GB) | $0 |
| Qdrant Cloud free tier (1 GB / 1M vectors) | $0 |
| Vercel Hobby plan | $0 |
| Groq Free tier (30 RPM, 14.4k req/day) | $0 |
| Hostinger Business plan | already paid |
| GitHub Actions cron | $0 (within 2000 min/mo free) |
| Custom domain | $0 (use `.hf.space` and `.vercel.app`) |
| **New spend** | **$0** |

## Out of scope

- Custom domain on HF Spaces (requires HF PRO at $9/mo) — skipped, `.hf.space` subdomain is acceptable for portfolio
- Phoenix observability in production (not free at scale)
- Postgres checkpointing (HF Space ephemeral, use SQLite wiped per session)
- GPU inference in production (HF Space CPU Basic is sufficient with current fine-tuned reranker)
- Multi-region failover (out of scope for free-tier demo)
- Email login / OAuth (BYOK demo has no accounts)

## Stop conditions

Halt the launch and escalate to owner if:

- Any signup in phase 1 asks for a credit card
- Any signup denies registration from Egypt
- Qdrant Cloud free tier drops sparse vector support (would break hybrid search)
- Groq adds a credit-card requirement to the free tier
- HF Space drops the 16 GB CPU Basic offering

Each stop condition has a fallback documented in `01-stack-decisions.md`.
