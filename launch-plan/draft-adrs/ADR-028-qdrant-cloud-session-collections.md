# ADR-028 (DRAFT): Qdrant Cloud Free Tier + Per-Session Collections

**Date:** 2026-05-25 (drafted) / TBD (accepted)
**Status:** **Draft** — flips to **Accepted** when the production backend reads/writes against Qdrant Cloud and the session-purge cron has run cleanly for 48 hours.

## Context

The platform uses Qdrant as its vector store, with the following non-negotiable features the production deployment must preserve:

- Dense + sparse hybrid (BM25 default, SPLADE opt-in)
- Per-tenant collections (ADR-021)
- RBAC payload filter applied to every search
- Sparse vector field with structural isolation across collections

For the HF Spaces backend (ADR-026), self-hosting Qdrant inside the Space is risky because the Space's disk is ephemeral — every container restart loses the vector data, which would surprise visitors mid-session. We need an externally-hosted Qdrant that stays alive across HF Space restarts.

Free-tier audit found:

| Option | Cost | CC required | Sparse vectors | Verdict |
|---|---|---|---|---|
| Qdrant Cloud free tier | $0 | no | yes | **chosen** |
| Pinecone free | $0 | no | indirect (separate index) | rejected (code change needed) |
| MongoDB Atlas pgvector | $0 | no | partial | rejected (different SDK) |
| Self-hosted in HF Space | $0 | n/a | yes | rejected (ephemeral disk) |
| Supabase pgvector | $0 | no | partial | rejected (different SDK) |

## Decision

Use **Qdrant Cloud free tier (1 GB / 1M vectors / always-on)** for the production vector store. Combine with **session-scoped collections** to deliver privacy + cleanup:

- Cluster: 1 free node in AWS us-east (low latency from Egypt to AWS US-East: ~150ms)
- Collection naming: `documents_sess_<sanitized_session_id>` per BYOK visitor
- TTL: 24 hours via `SAR_SESSION_TTL_HOURS`
- Purge cron: `retrieval/session_purge.py` runs every 6 hours via APScheduler inside the FastAPI lifespan
- Credentials: `SAR_QDRANT_URL` + `SAR_QDRANT_API_KEY` set as HF Space secrets
- Sparse vectors: SPLADE inference still runs on the HF Space CPU; sparse vector field provisioned in each per-session collection (same shape as ADR-024)

## Consequences

- (+) Always-on (no sleep affecting vector availability)
- (+) Zero migration — `retrieval/qdrant_client.py` already speaks to remote Qdrant URLs
- (+) Per-session isolation is structural (each session has its own collection, queries cannot reach across)
- (+) The 24h TTL is a clean answer to "what happens to my uploaded data" — public demo audiences want predictability
- (+) Same RBAC payload filter works because the filter is applied at query time, not at storage time
- (-) 1 GB cap limits the demo corpus to ~50k chunks total across all live sessions. Need to monitor and purge aggressively if traffic spikes
- (-) Network round-trip from HF Space → Qdrant Cloud adds ~30-50ms per request (acceptable, dwarfed by LLM latency)
- (-) Outbound traffic from HF Space to Qdrant Cloud is free for now; if HF Spaces adds egress fees in the future, this design must be revisited
- (-) Cluster URL embedded in Space secrets — rotating it means updating the Space config

## Alternatives considered

- **Self-host Qdrant in HF Space (same container)** — ephemeral disk would lose state on each restart. Rejected
- **Self-host Qdrant in HF Space with HF Persistent Storage** — costs $0.10/GB/mo. Violates zero-cost constraint. Rejected
- **Two separate Spaces (one Qdrant, one app)** — saves $$ but adds network hop + HF Spaces are not designed for headless data services. Rejected
- **Pinecone** — would require porting our code from Qdrant. Lost weekend. Rejected

## Acceptance criteria for flip to Accepted

- Qdrant Cloud free cluster provisioned and accessible from HF Space
- Phase 1 smoke test verifies sparse vector support on the cluster
- `retrieval/session_purge.py` runs successfully on a schedule and deletes only expired collections
- `tests/test_retrieval/test_session_purge.py` passes
- 48 hours of demo traffic without cluster capacity exceeded
- ADR-028 moved from `launch-plan/draft-adrs/` into `DECISIONS.md` with this content

## Related ADRs

- ADR-002 (Qdrant over Chroma/Pinecone) — the base decision; ADR-028 extends it for production
- ADR-021 (Multi-tenant collections via `for_org`) — session collections follow the same pattern
- ADR-024 (Per-tenant SPLADE manager cache) — session managers are cached the same way
- ADR-025 (BYOK demo mode) — the session UUID drives the collection name
- ADR-026 (HF Spaces backend) — the network topology motivates the external Qdrant choice
