# 00 — Context

## What this project is

SecureAgentRAG is a privacy-first, multi-agent, RAG platform built around four production patterns most demos skip:

1. **RBAC at the vector DB layer** — `org_id` + `sensitivity_level_int <= clearance` + `roles MATCH ANY` enforced as a Qdrant must-filter on every search. Structurally impossible to bypass from the application layer.
2. **Sensitivity-based inference routing** — HIGH-sensitivity documents force local Ollama inference; LOW can opt in to cloud Groq / OpenAI / Anthropic.
3. **NLI per-sentence faithfulness gate** — every cited sentence must be entailed by its source chunk under a natural language inference model. Citation marker presence is not enough.
4. **SHA-256 hash-chained audit log + SLO deadline** — every request gets a tamper-evident audit entry with PII redacted before persist.

## Repo state on `main` (frozen at `56c8c98`)

- 29.5k Python LOC across 147 files
- 487 tests passing, 22 skipped, 0 failed
- 24 ADRs (`DECISIONS.md`)
- 9-node LangGraph state machine (`core/graph.py`): router → guardrails → security → retriever → grader → rewriter → synthesizer → faithfulness → evaluator
- Three interfaces today: Streamlit (`app/main.py`), FastAPI (`interfaces/api.py`), MCP stdio (`interfaces/mcp_server.py`) — all share `core.schemas.QueryResponse`
- Qdrant native sparse vectors (BM25 default, SPLADE optional), per-tenant collection routing via `QdrantManager.for_org()`
- LlamaGuard 3 escalation backend for guardrails
- HS256 / RS256+JWKS JWT auth dispatch
- Fine-tuned reranker (P1, `+1.60pp NDCG@10` over BGE baseline) checkpoint at `data/checkpoints/reranker-domain-v1/`
- Calibrated confidence threshold at 0.35 (P2)
- 24/24 evidence grid PASS (`data/agent_evidence/`)

## What changes in this launch

| Layer | Today (main) | After launch (deploy/prod-launch) |
|---|---|---|
| Frontend | Streamlit on localhost | Next.js 15 + shadcn on Vercel |
| Backend | FastAPI on localhost via uvicorn | FastAPI in Docker on HF Space port 7860 |
| Vector store | Local Qdrant in docker-compose | Qdrant Cloud free tier (1 GB / 1M vectors) |
| LLM | Ollama (qwen3:8b) locally + optional Groq | Groq owner-key (rate-limited) + visitor BYOK |
| Embeddings | bge-m3 via Ollama | bge-m3 via sentence-transformers in HF Space RAM |
| Audit | SQLite on disk, hash-chained | SQLite in HF Space ephemeral disk, wiped per session |
| Auth | JWT (HS256 dev / RS256 prod) | None for demo (BYOK is the auth model) |
| Persona switching | Streamlit sidebar | Next.js dropdown |
| Domain | `localhost` | `*.hf.space` + `*.vercel.app` + `<hostinger-subdomain>` landing |

## What does NOT change

- The 9-node LangGraph remains identical
- All security primitives (RBAC filter, faithfulness gate, audit chain, sensitivity router) remain identical
- All 487 existing tests must still pass on `deploy/prod-launch`
- The fine-tuned reranker, calibration, multi-tenant collections, all unchanged
- `interfaces/api.py` gains a BYOK mode but its existing endpoints remain backward compatible

## Why this stack (one line each)

- **Hugging Face Spaces** — only no-CC, Egypt-OK, Docker-supporting, 16 GB RAM free tier in 2026 that sleeps after 48 hours not 15 minutes
- **Qdrant Cloud free** — same SDK we already use; sparse vectors and RBAC filter port unchanged
- **Vercel Hobby** — free, no CC, no cold start for Next.js SSR; standard portfolio expectation
- **Groq** — already integrated, owner key already in `.env`, no CC required for free tier
- **Hostinger Business Web Hosting** — already paid by owner, fine for static landing only (no FastAPI on shared)

## Repo personnel / handoff

- **Owner:** Moaz Muhammad (`moazmo27@gmail.com`, GitHub `moazmo`)
- **Commits must show owner identity** — no AI co-author footers (per global instructions in `~/.claude/CLAUDE.md`)
- **Private files** the agent has access to and must keep updated: `.env`, `INTERVIEW_DEFENSE.md`, `CV_BLURB.md`, `private/roadmap.md`, `NOTES.md`, `TODO.md`
- **Files that must NEVER be modified outside their normal flow:** `audit_logs/*.jsonl` (hash chain), `data/checkpoints.sqlite` (LangGraph state), `uv.lock` (only via `uv` commands), `data/checkpoints/reranker-domain-v1/` (2.27 GB safetensors, gitignored)

## Reading order if you are an AI agent picking this up

1. This file (`00-context.md`)
2. `01-stack-decisions.md`
3. Run `git status` and `git log --oneline -10` — confirm you are on `deploy/prod-launch`
4. `02-smoke-tests.md` — check phase 1 boxes. If not all green, stop and escalate
5. Phase you are starting (`03` through `08`)
6. `11-security-checklist.md` before any code that touches keys or audit
7. `12-agent-handoff.md` — the contract you operate under
