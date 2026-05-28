# ADR-025 (DRAFT): BYOK Demo Mode

**Date:** 2026-05-25 (drafted) / TBD (accepted)
**Status:** **Draft** — flips to **Accepted** when `interfaces/byok.py` lands and `tests/test_security/test_byok_key_redaction.py` is green.

## Context

We want a public demo where any visitor — recruiter, peer, hiring manager — can try the platform without us paying per-token LLM costs. Three patterns existed:

1. **Owner-funded** — single Groq key paid by owner. Visitors fire arbitrary queries. Free tier exhausts in hours under any traffic. Rejected.
2. **Per-user OAuth + cloud credits** — sign visitors up, give them N free queries. Friction kills the demo. Recruiter spends 30 seconds clicking through OAuth before they see anything. Rejected.
3. **BYOK — visitor pastes their own key** — zero key-burn risk, but introduces a new threat surface: the visitor's key travels through our backend. Must never persist.

Independent constraint: HF Spaces is the chosen backend host (see ADR-026). HF Spaces ephemeral disk + 48-hour sleep means we cannot rely on durable state. Forces a stateless-per-session design that matches BYOK naturally.

## Decision

Implement a BYOK mode behind `SAR_BYOK_MODE=true`. Concretely:

- Per-request headers `X-User-LLM-Key`, `X-User-Provider`, `X-User-Ollama-URL`, `X-Session-ID` extracted by a FastAPI dependency
- Owner-key fallback gated by a per-IP rate limit (3 queries/hour default, configurable via `SAR_BYOK_OWNER_QUOTA`)
- Each session gets a Qdrant collection named `documents_sess_<sanitized_session_id>` — uploaded PDFs are isolated to that session
- A purge cron deletes collections older than `SAR_SESSION_TTL_HOURS` (default 24)
- Phoenix instrumentation forcibly disabled when `SAR_BYOK_MODE=true`
- Audit log redacts API key patterns (Groq `gsk_*`, OpenAI `sk-*`, Anthropic `sk-ant-*`) before persist
- CORS allowlist limited to the Vercel frontend URL (`SAR_CORS_ALLOW_ORIGINS`)
- localStorage on the frontend stores the BYOK key — never cookies (CSRF surface)

## Consequences

- (+) Zero recurring LLM cost regardless of demo traffic
- (+) Recruiter can paste a $0.10-test-budget key and verify the entire stack works without giving us anything
- (+) Per-session collection isolation is a free side-effect that doubles as multi-tenancy proof
- (+) Aligns with the project's privacy-first narrative — "we never store your keys, even for telemetry"
- (-) Introduces a new threat surface — must add and maintain key redaction tests
- (-) Owner-key fallback still exists for visitors without a key, costing rate-limit budget; we throttle aggressively
- (-) Session cleanup adds a cron job we have to keep alive
- (-) CORS allowlist must be updated whenever the Vercel frontend URL changes

## Alternatives considered

- **Force BYOK only (no owner fallback)** — would feel hostile to first-time visitors. Owner-key fallback with 3/hour limit is the right UX
- **Charge owner-funded with strict per-IP budget** — needs CC on Groq, increases risk surface, doesn't scale narrative-wise
- **Run everything client-side via WebLLM** — would lose the entire RAG stack. Rejected

## Acceptance criteria for flip to Accepted

- `interfaces/byok.py` ships
- `SAR_BYOK_MODE=true uvicorn interfaces.api:app` boots without error
- `tests/test_security/test_byok_key_redaction.py` passes (Groq + OpenAI + Anthropic keys redacted)
- `tests/test_interfaces/test_byok.py` passes (per-IP throttle, header extraction, session ID generation)
- `tests/test_retrieval/test_session_purge.py` passes (cron deletes only expired collections)
- ADR-025 moved from `launch-plan/draft-adrs/` into `DECISIONS.md` with this content

## Related ADRs

- ADR-026 (HF Spaces as backend host) — provides the ephemeral context that motivates session isolation
- ADR-027 (Vercel + Next.js frontend) — provides the BYOK input UX
- ADR-028 (Qdrant Cloud + session collections) — provides the per-session storage mechanism
- ADR-021 (Multi-tenant collections) — BYOK builds on top of the existing per-tenant routing
