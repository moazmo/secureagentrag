# ADR-027 (DRAFT): Vercel + Next.js 15 Frontend (drop Streamlit for public demo)

**Date:** 2026-05-25 (drafted) / TBD (accepted)
**Status:** **Draft** — flips to **Accepted** when the Vercel deployment is live and a manual smoke from Egypt streams a real response.

## Context

The platform ships a Streamlit UI (`app/`) that has been the primary interface throughout development. For the public production launch we need a frontend that:

- Streams tokens with low latency (acceptable response on slow connections)
- Renders well on mobile (recruiters may visit from phones)
- Looks professional in 2026 standards (shadcn/ui patterns, not Streamlit's data-science aesthetic)
- Hosts on a no-CC free tier with no cold start
- Lets visitors paste their own LLM API key (BYOK — see ADR-025)
- Doesn't require us to host a long-running Python server (saves the HF Space CPU budget for the backend)

Streamlit on HF Spaces is technically possible but:

- Recruiters discount Streamlit as "data-scientist tooling" — visible signal mismatch with a production-grade narrative
- Streamlit websockets struggle through HF's proxy in edge cases (auto-rerun reconnects)
- Streamlit is not built around BYOK as a first-class pattern; we'd be working against the grain
- A single-container deployment (Streamlit + FastAPI on the same Space) loses the architectural clarity we want to communicate

## Decision

Build a **Next.js 15 App Router frontend with shadcn/ui + Tailwind v4 + Vercel AI SDK**, deployed to **Vercel Hobby plan**. Streamlit remains in the repo for local development but is not the public face of the demo.

- Repo: separate sibling repo `secureagentrag-web` at `github.com/moazmo/secureagentrag-web`
- Deploy: Vercel free Hobby plan
- URL: `app.eilm.live`
- Streaming: Vercel AI SDK `useChat` hook over SSE bridged to the FastAPI streaming endpoint
- BYOK input: drawer + localStorage persistence (never cookies)
- Audit viewer: client-side hash verification
- Persona switcher: three preset RBAC profiles (engineer / compliance / executive)
- PDF upload: `react-dropzone` → POST to backend
- Theme: dark-first neutral palette

## Consequences

- (+) Zero cold start (Vercel Edge for SSR, Edge function for streaming bridge)
- (+) Industry-standard portfolio signal — "Next.js on Vercel" matches recruiter mental models for AI products
- (+) BYOK UX is clean — drawer + localStorage is a familiar pattern
- (+) Vercel AI SDK eliminates a week of streaming-state-machine plumbing
- (+) Mobile responsiveness comes free from shadcn/Tailwind primitives
- (+) Separates frontend and backend lifecycles — can iterate on UI without redeploying the model layer
- (-) Two repos to maintain instead of one
- (-) Some duplication of types between TypeScript (frontend) and Python (backend); mitigated by `openapi-typescript` codegen against the FastAPI schema
- (-) Streamlit on `main` becomes the "local dev demo" rather than the public demo. Renaming/clarifying needed in `app/`
- (-) Streaming wire format needs an SSE adapter between Vercel AI SDK conventions and our `graph.astream(stream_mode=["updates","custom"])` events. ~80 LOC of bridge code

## Alternatives considered

- **Keep Streamlit on HF Space** — recruiter optics + Streamlit-on-HF reliability concerns. Rejected
- **Build Next.js but host on HF Space alongside FastAPI** — would cost CPU budget on the backend, loses the architectural separation, loses edge caching
- **Use Vercel's Python runtime for backend too** — Vercel Python functions cap at 60s on Hobby. Our pipeline can take longer. Rejected
- **SvelteKit on Cloudflare Pages** — lighter but smaller ecosystem; Vercel AI SDK is the strongest fit for streaming LLM UIs in 2026

## Acceptance criteria for flip to Accepted

- `secureagentrag-web` repo created on GitHub
- Vercel project linked
- `vercel --prod` deploys cleanly
- `https://app.eilm.live` reachable from Egypt
- BYOK drawer saves to localStorage and forwards header on next request
- End-to-end streaming smoke against live HF Space backend works
- Lighthouse mobile ≥ 90 on Performance, Accessibility, Best Practices, SEO
- ADR-027 moved from `launch-plan/draft-adrs/` into `DECISIONS.md` with this content

## Related ADRs

- ADR-025 (BYOK demo mode) — the frontend hosts the BYOK input UX
- ADR-026 (HF Spaces backend) — the frontend talks to the HF Space via fetch
- ADR-028 (Qdrant Cloud) — the frontend's session UUID maps to a Qdrant collection
