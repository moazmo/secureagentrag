# CLAUDE.md — Project Brief for AI Coding Agents

This file is the canonical entry point for any AI agent (Claude / Hermes / Kimi / Cursor / Aider) picking up work on SecureAgentRAG. Read it before touching code.

> 🚀 **Launch complete + hardened — `main` is the trunk.** The production BYOK demo (P6) shipped and merged: `deploy/prod-launch` → `main` on 2026-05-28, tagged **`v1.0.0-launch`**, CI green. Since launch, five post-launch hardening waves landed on `main` (ADR-031..033 + a production rate-limit fix): **Wave 2** Prometheus/Grafana observability; **Wave 3** auth fail-closed + OCR off the event loop + scheduled audit-chain verify + frontend security headers; **Wave 4** batched NLI faithfulness + real-Qdrant CI job; **Wave 5** Node-24 CI + selective guardrail escalation; **rate-limit fix** streaming 429 retry/backoff + `SAR_SYNTH_MAX_TOKENS` cap + honest copy. New work goes on `main` directly or a feature branch. `private/roadmap.md` holds the history; `private/review-2026-05-28.md` is the launch-era deep review.

---

## 1. What the project is

**SecureAgentRAG** — privacy-first, multi-agent RAG platform.

The hero story is four things most RAG demos skip:

1. **RBAC at the vector-DB layer** — Qdrant payload filters block unauthorized docs regardless of similarity. Multi-tenant collections layer on top via `QdrantManager.for_org()`.
2. **Sensitivity-based inference routing** — HIGH-classified data physically never leaves local Ollama. Cloud is opt-in for LOW/MEDIUM only.
3. **Corrective RAG + NLI faithfulness gate** — Cited sentences are re-checked for entailment against the source chunk. Unsupported claims are flagged or dropped.
4. **Tamper-evident audit chain + SLO deadline** — SHA-256 hash chain across JSONL entries; `SAR_REQUEST_TIMEOUT_S` bounds the whole pipeline.

Owner: `moazmo` / `moazmo27@gmail.com`. License: MIT.

---

## 2. Stack

| Layer | Choice | File |
|---|---|---|
| Orchestration | LangGraph 1.x StateGraph, 9 nodes | `core/graph.py`, `core/state.py` |
| Vector DB | Qdrant (docker, `:6333` REST) | `retrieval/qdrant_client.py` |
| Local LLM | Ollama (`qwen3:8b` Q4_K_M) | `inference/ollama_client.py` |
| Embeddings | BGE-M3 multilingual via Ollama | `retrieval/embeddings.py` |
| Sparse | Qdrant native sparse vectors — `bm25` (default) / `splade` | `retrieval/sparse_embeddings.py`, `retrieval/hybrid_search.py` |
| Rerankers | `none` / `cross_encoder` / `colbert` / `fine_tuned` | `retrieval/reranker.py`, `retrieval/colbert_reranker.py`, `core/agents/retriever.py::_get_reranker` |
| Cloud LLMs | Groq / OpenAI (shared `OpenAICompatibleClient`) / Anthropic | `inference/cloud_clients.py` |
| Persistence | AsyncPostgres → AsyncSqlite → MemorySaver | `core/graph.py::_get_*_checkpointer` |
| UI | Streamlit (sidebar + 4 tabs) | `app/main.py`, `app/views/*`, `app/chat_service.py` |
| API | FastAPI (`:8080`) + MCP stdio | `interfaces/api.py`, `interfaces/mcp_server.py` |
| Auth | HS256 (default) / RS256 + JWKS dispatch | `utils/auth.py`, `utils/jwks_cache.py` |
| Guardrails | regex → `llm` → `llamaguard` (S1-S14 taxonomy) | `core/agents/guardrails.py`, `core/agents/guardrails_llm.py`, `core/agents/guardrails_llamaguard.py` |
| Faithfulness | Per-sentence NLI entailment gate | `core/agents/faithfulness.py` |
| Observability | structlog + Arize Phoenix tracing + Prometheus `/metrics` → Grafana | `utils/logging.py`, `utils/observability.py`, `utils/metrics.py`, `deploy/grafana/` |
| Audit auto-verify | Scheduled SHA-256 chain re-verification + metric (FastAPI lifespan) | `utils/audit_verify.py` |
| Eval | Ragas + custom metrics | `evaluation/*` |
| Audit | SHA-256 hash chain, JSONL, PII-redacted | `utils/audit.py`, `utils/pii.py` |
| Package mgr | uv | `pyproject.toml`, `uv.lock` |
| Python | 3.11 (pinned `>=3.11,<3.14`) | — |

---

## 3. Project layout

```
secureagentrag/
├── app/                # Streamlit UI
│   ├── main.py
│   ├── views/          # chat / upload / audit / evaluation
│   └── components/     # sidebar + chat_message
├── core/
│   ├── graph.py        # LangGraph compile + run_rag_pipeline[_stream]
│   ├── state.py        # GraphState TypedDict + reducers
│   ├── schemas.py      # Pydantic v2 request/response schemas
│   └── agents/         # 9 nodes
├── retrieval/
│   ├── qdrant_client.py        # RBAC filter, multi-tenant for_org, sparse upsert/search
│   ├── hybrid_search.py        # dense + Qdrant native sparse + RRF, one RBAC filter shared
│   ├── sparse_embeddings.py    # bm25 / splade backends, SparseVector emitter
│   ├── embeddings.py
│   ├── reranker.py / colbert_reranker.py
│   ├── self_query.py           # NL → structured filter
│   ├── hyde.py
│   └── multitenancy.py         # get_collection_name
├── ingestion/
│   ├── pipeline.py
│   ├── chunker.py          # Arabic-aware
│   ├── loaders.py
│   ├── ocr.py / vlm_ocr.py # PaddleOCR + Qwen-VL primary
│   └── multimodal.py
├── inference/
│   ├── router.py           # sensitivity-gated routing
│   ├── llm_factory.py
│   └── cloud_clients.py
├── interfaces/
│   ├── api.py              # FastAPI + /token endpoint
│   └── mcp_server.py
├── app/chat_service.py     # extracted audit + ragas eval helpers
├── core/agents/
│   ├── faithfulness.py     # NLI per-sentence entailment gate
│   ├── guardrails.py       # regex + backend dispatch
│   ├── guardrails_llm.py   # legacy SAFE/UNSAFE escalation on qwen3:8b
│   └── guardrails_llamaguard.py  # llama-guard3:8b (S1-S14 taxonomy)
├── evaluation/             # ragas, cost dashboard, benchmark, nightly
├── utils/                  # auth, jwks_cache, audit, pii, logging, rate_limiter, ...
├── scripts/                # smoke, seed_corpus, interview_demo, quick_bench,
│                           # cloud_bench, h2_gate, migrate_to_splade,
│                           # train_reranker, bench_reranker, verify_audit_chain
├── tests/                  # pytest, 656 unit + 2 live-Qdrant integration (test_qdrant_rbac)
├── helm/secureagentrag/    # Kubernetes manifests
├── deploy/                 # docker-compose auth profile + keycloak-realm.json
├── data/agent_evidence/    # 24-scenario gate evidence (results.md, screenshots)
├── sample_docs/            # PDF + txt corpus (incl. real NIST AI RMF)
├── DECISIONS.md            # ADR-001..033
├── docker-compose.observability.yml  # Prometheus + Grafana overlay (self-hosted)
├── architecture.md         # Mermaid diagrams
├── RUNBOOK.md              # ops + troubleshooting
├── README.md               # public face
└── INTERVIEW_DEFENSE.md    # gitignored personal notes
```

---

## 4. Commands you will actually run

```bash
uv sync                                              # install
uv sync --extra api --extra persistence --extra all  # install everything

# Tests / lint / format — must stay green before any commit
uv run pytest -q
uv run ruff check .
uv run ruff format --check .

# Run the UI
docker compose up -d qdrant postgres
ollama pull qwen3:8b && ollama pull bge-m3
uv run streamlit run app/main.py

# Smoke + interview demo
uv run python -m scripts.seed_corpus --mode rbac
uv run python -m scripts.e2e_smoke
uv run python -m scripts.interview_demo
uv run python -m scripts.verify_audit_chain

# Bench
uv run python -m scripts.quick_bench          # local
uv run python -m scripts.cloud_bench --quick  # cloud-only, ~2 min
uv run python -m scripts.cloud_bench          # local + cloud comparison
```

---

## 5. State of the codebase (as of `3ad56bb` on `main`)

- **656 unit tests pass + 2 live-Qdrant integration tests.** 0 failed. Lint + format clean. **CI green on `main`** — now **two jobs**: a unit job (`uv sync --frozen --group dev --extra api` → ruff check + ruff format --check + `pytest -m "not integration"`) and an **integration job** that spins up a `qdrant/qdrant` service container and runs `pytest -m integration` (proves the RBAC filter against a real Qdrant). GitHub Actions on Node 24 (checkout v5 / setup-python v6 / setup-uv v6).
- **~35.5k Python LOC** across 169 files.
- **33 ADRs** in `DECISIONS.md` (001–024 historical + 025–030 the launch + **031** Prometheus/Grafana metrics + **032** security/reliability hardening + **033** cost/coverage hardening).
- **Observability:** structlog logs + optional Phoenix tracing + **Prometheus `/metrics` (`utils/metrics.py`) → Grafana dashboard (`deploy/grafana/`, `docker-compose.observability.yml`)**. Metrics are aggregate-only (no prompt/key/user text in labels) so they are BYOK-safe; Phoenix tracing stays hard-disabled under BYOK. The FastAPI lifespan (`interfaces/api.py`) starts a scheduled **audit-chain re-verification** (`utils/audit_verify.py`, emits `audit_chain_valid`) and wires the BYOK session-purge job.
- **Auth fails closed:** with no `SAR_JWT_SECRET`, every bearer token is rejected unless `SAR_ALLOW_UNSIGNED_TOKENS=true` (dev/test only). The legacy unsigned base64 shape is never accepted silently (ADR-032).
- **Launch merged to `main`** 2026-05-28 (merge `e6f2507`), tagged **`v1.0.0-launch`**. 25 commits past the old frozen point `56c8c98`. The freeze is lifted — `main` is the trunk.
- **Live $0/month production deploy:**
  - Frontend: `https://secureagentrag-web.vercel.app` — Next.js 16 + Tailwind v4 + SSE streaming on Vercel Hobby
  - Backend: `https://LeomordKaly-secureagentrag-api.hf.space` — FastAPI BYOK on HF Spaces Docker CPU Basic, 16 GB RAM, 48 h sleep defeated by GitHub Actions cron
  - Vector store: Qdrant Cloud free tier (1 GB, AWS us-east-1) — 10 demo RBAC docs (138 chunks, 276 points incl. sparse) + per-session collections
  - LLM: Groq free tier (`llama-3.1-8b-instant`, 14,400 RPD, 30 RPM, per-IP owner throttle + visitor BYOK unlock)
- **9 graph nodes:** router → guardrails → security → retriever → grader → rewriter → synthesizer → faithfulness → evaluator. Untouched structurally. In BYOK mode several nodes bypass their LLM call for cost (see ADR-030).
- Streamlit, FastAPI, MCP all share `core.schemas.QueryResponse`. Streamlit is now the **local dev face**; Next.js is the **public production face**.
- **Hybrid search** uses Qdrant native sparse vectors (BM25 default in production). Cross-tenant + cross-session bypass is structurally impossible.
- **Auth** dispatches HS256 ↔ RS256 + JWKS via `SAR_JWT_ALGORITHM`. The BYOK demo does not require auth (it has its own session/persona contract).
- **Guardrails** in production: regex only (LlamaGuard escalation is off in BYOK mode to save Groq calls). Self-hosted deploys flip `SAR_GUARDRAILS_BACKEND=llamaguard` to restore S1-S14 taxonomy.
- **Reranker** factory accepts `none` / `cross_encoder` / `colbert` / `fine_tuned`. Live production uses `none` (CPU Basic disk budget + small corpus); fine-tuned checkpoint trained on RTX 3060 (+1.60pp NDCG@10) lives at `data/checkpoints/reranker-domain-v1/` (gitignored).
- **Threshold calibration** — `evaluation/calibration.json` loaded at import via `_apply_calibration`. Env override still wins.
- **24-scenario UI gate (H.1 + H.2) is 24/24 PASS** on `main`. Live BYOK demo verified end-to-end with kubectl rollback chat returning full citations in 13 s.
- Recent commits (chronological, newest first):
  - `3ad56bb` / `582bd86` fix(byok): streaming 429 retry + backoff + Retry-After; `SAR_SYNTH_MAX_TOKENS` cap; honest per-minute copy (production rate-limit fix)
  - `6191848` / `22a4765` test(ci): real-Qdrant RBAC integration job + batched NLI faithfulness (ADR-033)
  - `82567c2` / `4da45af` / `306aae8` Wave 3: auth fail-closed + OCR `to_thread` + scheduled audit verify + frontend security headers (ADR-032)
  - `49216cf` / `e097443` feat(observability): Prometheus `/metrics` + Grafana dashboard (ADR-031)
  - `e848ecf` fix(byok): cut Groq RPM pressure by 50% — pin 8b-instant, kill RAG-fusion, bypass evaluator LLM, router shortcut (ADR-030)
  - `7b6997a` fix(byok): drop sensitivity disclaimer + actionable LLM-fail copy (W-series)
  - `ac61654` fix(byok): upload + chat quality hardening for free-tier Groq (V-series)
  - `1c8c7ad` feat(byok): visitor document upload with dual-collection retrieval (ADR-029, U-series)
  - `1ae38dc` feat(byok): streaming SSE + audit export + persona prompts + XFF + HIGH unlock (A-series)
  - `cd4bc80` docs(smoke): drop key-rotation TODO from Groq smoke
  - `f7b455d` revert(landing): drop Hostinger landing + `app.eilm.live` custom domain
  - `d8cb15f` deploy(corpus): phase 7 — ingest 10 demo docs + RBAC live
  - `9bc9526` ci(keepalive): daily cron pings HF Space + Vercel
  - `1e80b8b` deploy(vercel): phase 4 frontend live (ADR-027)
  - `9a4e3ea` deploy(hf): phase 3 backend live (ADR-026)
  - `9b430cc` feat(byok): phase 2 backend BYOK mode — `/byok/chat` live, 113 new tests (ADR-025)
  - `5edc858`/`7890234`/`05e6d6e`/`cb3cdd3` deploy: phase 1 smoke signups (HF + Vercel + Groq + Qdrant Cloud + Hostinger inventory)
  - `d6625d8` docs(launch): P6 plan + handoff contract
  - `abe100e` fix(ci): install api extra so the BYOK + REST suite runs (623 tests gate CI)
  - `ffa23d0` fix(ci): clear ruff lint + format failures surfaced by the first main CI run
  - `e6f2507` merge: production launch P6 + answer-quality + transparency (**v1.0.0-launch**)
  - `8a1eab6` *(web)* feat: Markdown answers + in-chat knowledge base + OG image + analytics
  - `3ffd311` feat(synth): richer Markdown answers + 1100-char per-chunk context
  - `d8639f6` feat(byok): public corpus + personas metadata endpoints + full doc sweep
  - `56c8c98` *(old pre-launch baseline)* feat(retrieval): cache per-tenant QdrantManagers + pin sparse isolation (ADR-024)

---

## 5.1 BYOK production mode (the live demo's runtime contract)

Lives behind `SAR_BYOK_MODE=true`. The HF Space Dockerfile sets it; local dev does not.

- **Request shape:** `X-Demo-Persona` (engineer / compliance / executive — preset RBAC), `X-Session-ID` (UUID, drives session collection), `X-User-LLM-Key` (optional — visitor BYOK unlock), `X-User-Provider` (groq / openai / anthropic), `X-User-Ollama-URL` (optional). Extracted in `interfaces/byok.py`.
- **Endpoints (under `/byok/`):** `chat` (sync JSON), `chat/stream` (SSE: open|phase|token|blocked|final|error), `audit` (last-N session-scoped rows for export), `uploads` GET/POST/DELETE (5 MB · 5 files · 60 chunks/file · txt/md/pdf — see ADR-029), `personas` (RBAC dispatch table, no auth), `corpus` (base demo corpus metadata, no auth — never returns chunk text).
- **Per-IP throttle:** `SAR_BYOK_OWNER_KEY_QUOTA_PER_HOUR=10` against owner key. Visitor BYOK bypasses. IP from `X-Forwarded-For` leftmost token (HF Spaces reverse proxy masks `request.client.host`).
- **Persona presets:** `_DEMO_PERSONAS` in `interfaces/api.py` maps each persona to `(clearance, roles, style)`. Style is threaded into the synth system prompt via `GraphState.persona_style`.
- **Cost-cut toggles (ADR-030):** `SAR_GROQ_MODEL=llama-3.1-8b-instant`, `SAR_RAG_FUSION_ENABLED=false`, `SAR_BYOK_SKIP_EVALUATOR=true`, `SAR_BYOK_SKIP_GRADER=true`, `SAR_FAITHFULNESS_GATE_ENABLED=false`, `SAR_RERANKER_TYPE=none`, `SAR_RELEVANCE_THRESHOLD=0.55`, `SAR_MAX_RETRIES=1`, `SAR_RERANK_TOP_K=10`. Router classifier short-circuits queries ≤80 chars to `query_type="simple"`. Net effect: ~2 Groq calls/chat vs ~5–6 before.
- **Rate-limit hardening (post-launch fix):** `SAR_SYNTH_MAX_TOKENS=1024` on the Space caps synth completion tokens to ease the Groq 6k TPM ceiling; the streaming cloud client retries a 429 with backoff **before the first token** (honors `Retry-After`) so a transient per-minute limit no longer kills the answer. The "shared key per-minute limit" copy is honest (not "exhausted for the hour"). See `inference/cloud_clients.py::_stream_lines_with_retry` + `core/agents/router.py::call_llm_stream`.
- **HIGH-on-cloud unlock:** `SAR_ALLOW_CLOUD_FOR_HIGH=true` in production because the HF Space has no Ollama. Frontend renders a `sensitivity:` badge so the visitor is informed. *Hero claim "HIGH never leaves local" is true in self-hosted mode only.*
- **Session collections:** `documents_sess_<sanitized_session_id>`. Dual-collection retrieval (base ∪ session) under one RBAC filter; RRF-fused. 24 h TTL via `SAR_SESSION_COLLECTION_TTL_HOURS=24` — `retrieval/session_purge.py::purge_expired_sessions` runs every 6 h via `schedule_session_purge` in the FastAPI lifespan (APScheduler).
- **Audit:** session-scoped only (`/byok/audit` filters `user_id == "demo-<sid>"`). SHA-256 chain intact; downloadable JSONL.
- **Sensitivity disclaimer suppressed** in BYOK mode (both prompt-side gate in `_build_system_prompt` and post-synth `_add_disclaimers` early-return). The frontend's `sensitivity:` badge is the user-facing signal.
- **No Phoenix / Postgres / Ollama in BYOK mode.** Audit on /tmp; checkpointer in-memory.

## 6. Genuinely remaining work (audited 2026-05-28)

The BYOK launch is complete, merged to `main`, tagged `v1.0.0-launch`, CI green. Follow-up quality + transparency pass shipped (Y-series web product surface, Z-series Markdown answers + in-chat knowledge base). **101 s demo video shipped** (Remotion in `secureagentrag-video/`, on the release + inline in both READMEs). Pipeline survives Egypt-from-mobile traffic on $0/mo. **No launch items open.** Optional polish only:

- **Optional:** enable Vercel Web Analytics + Speed Insights in the dashboard (code already wired in `secureagentrag-web` layout — no-op until toggled).
- **Optional:** upload fine-tuned reranker to `LeomordKaly/secureagentrag-reranker-v1` HF Hub model repo and flip `SAR_RERANKER_TYPE=fine_tuned` on the Space. **Not recommended** on the 10-doc corpus — ADR-022/030 bench shows the cross-encoder's top-5 cut drops the visitor's own chunk; helps only past ~200 docs/query.
- **Optional:** wire SonarCloud quality gate (currently "not computed" — neutral, non-blocking).
- **Optional (definitive concurrency fix):** put a **paid Groq key** as the Space owner key (`SAR_GROQ_API_KEY`) — 10×+ limits remove the residual simultaneous-heavy-user rate limit. Pure env change. Until then BYOK is the unlimited path.

**Recently shipped** (was in this section, now done):
- ✅ **Prometheus/Grafana observability** — `utils/metrics.py` + `deploy/grafana/` + `docker-compose.observability.yml`; aggregate-only, BYOK-safe. ADR-031.
- ✅ **Security/reliability hardening** — auth fail-closed (`SAR_ALLOW_UNSIGNED_TOKENS`), OCR off the event loop (`asyncio.to_thread` in `ingestion/pipeline.py`), scheduled audit-chain verify (`utils/audit_verify.py` on lifespan), frontend security headers (`next.config.ts` CSP/HSTS/etc.). ADR-032.
- ✅ **Batched NLI faithfulness** — `SAR_FAITHFULNESS_BATCH_ENABLED` (default on, size 8); N calls → ceil(N/8) with per-claim fallback. ADR-033.
- ✅ **Real-Qdrant RBAC integration test + CI job** — `tests/test_integration/test_qdrant_rbac.py`, dedicated CI job with a Qdrant service container. ADR-033.
- ✅ **Selective guardrail escalation** — strict mode escalates to LlamaGuard only on *suspicious* queries (`SAR_GUARDRAILS_SELECTIVE_ESCALATION`, default on). ADR-033.
- ✅ **GitHub Actions on Node 24** — checkout v5 / setup-python v6 / setup-uv v6. ADR-033.
- ✅ **Streaming rate-limit fix** — `inference/cloud_clients.py` retries 429 before the first token (honors `Retry-After`); `SAR_SYNTH_MAX_TOKENS` caps TPM; honest per-minute copy in `core/agents/router.py`.
- ✅ **Per-tenant SPLADE manager cache** — `QdrantManager.for_org(org_id)` now caches per-tenant managers; cross-tenant sparse isolation is pinned by 3 new regression tests. ADR-024 (2026-05-23).
- ✅ **Chat view slim** — `app/views/chat.py` 621 → 161 LOC. Streaming / sync / sidebar / persist extracted into focused modules.
- ✅ **NIST in-domain rerank gold** — `evaluation/nist_rerank_gold.jsonl` (20 hand-picked triplets from the NIST AI RMF corpus). Unlocks the NIST arm of `scripts/bench_reranker.py`. Current run: candidate beats baseline by **+0.54pp NDCG@10** (0.9162 → 0.9215). ADR-022 acceptance criteria fully met.
- ✅ **Threshold calibration shipped** — 50-row labelled gold set + `scripts/calibrate_thresholds.py` sweep; `evaluation/calibration.json` consumed by `config/settings.py::_apply_calibration` at import. Nightly CI gates on the measured baseline emitted by the same run. ADR-023 (2026-05-23).
- ✅ **Reranker fine-tune trained** — +1.60pp NDCG@10 vs BGE-Reranker-v2-M3 on MS-MARCO 500-pair hold-out (0.7744 → 0.7904). RTX 3060, 100k rows, 1 epoch, AMP fp16, ~4 hr wall. Bench: `evaluation/benchmarks/reranker_finetune.md`. ADR-022 status → fully Accepted (2026-05-23).
- ✅ Qdrant native sparse vectors (ADR-020, commit `26500ae`)
- ✅ RS256 + JWKS auth (ADR-019, commit `e80a519`)
- ✅ LlamaGuard 3 escalation backend (ADR-021, commit `038fdae`)
- ✅ Fine-tuned reranker scaffolding (ADR-022, commit `2f0e28d`)
- ✅ Cloud-client consolidation (-203 LOC, commit `45ebfde`)
- ✅ Chat-service extraction (-162 LOC, commit `6722772`)
- ✅ 24-scenario UI gate H.1 + H.2 (24/24 PASS, commit `1bcde26`)

---

## 7. Quality bar (non-negotiable)

These rules apply to every change, no exceptions:

1. **Less code beats more code.** Prefer deletion. If a feature can be expressed in fewer LOC without losing clarity, do it. PRs that grow LOC must justify why.
2. **Tests stay green.** `uv run pytest -q` → 0 failures. Add tests for new code.
3. **Lint + format clean.** `uv run ruff check .` and `uv run ruff format --check .` both pass.
4. **Real data over toy data.** Where a feature affects retrieval, training, or eval — drive it on the bundled NIST AI RMF corpus or a HuggingFace dataset (MS-MARCO, TREC-COVID). No more `["foo", "bar", "baz"]` fixtures for anything that touches retrieval quality.
5. **Defense in depth.** Security features (RBAC, guardrails, auth, audit) require a regression test that demonstrates the failure mode being closed.
6. **Audit-first.** Every node writes to `audit_trail`. Every API call lands in the SHA-256-chained audit log via `utils.audit`. PII is redacted before persistence.
7. **Fail closed on classification, fail open on transport.** A broken security LLM blocks the query. A transient embedding error degrades to BM25-only.
8. **Provenance everywhere.** `synth_provider`, `synth_model`, `forced_local`, `routing_reason`, `jti` all land in the audit trail.
9. **Backwards compat is documented in commits.** Breaking changes get a stanza in `DECISIONS.md` (next ADR number).
10. **Never commit secrets.** `.env` is gitignored. `INTERVIEW_DEFENSE.md` is gitignored. Never push API keys.

---

## 8. Files agents should NOT touch without explicit human approval

- `.env` (secrets)
- `INTERVIEW_DEFENSE.md` (personal interview prep, gitignored)
- `pyproject.toml` dependency pins — when adding a dep, run `uv add <pkg>` and commit the lock change.
- `uv.lock` — only modified by `uv` commands.
- `audit_logs/*.jsonl` — hash-chain integrity will break if touched.
- Anything under `data/checkpoints.sqlite` — LangGraph thread state.

---

## 9. Conventions

- **Commit messages:** Conventional Commits. Subject ≤72 chars, body ≤80-col wrap, second-person voice. No AI attribution. See git log for shape.
- **Pull requests:** title is a one-line summary; body has Summary + Test plan + Files. No AI attribution.
- **Python style:** ruff-enforced. Type hints everywhere. `from __future__ import annotations` at the top of new files. Pydantic v2 for domain models, TypedDict for graph state.
- **Async:** every node is `async def node(state: GraphState) -> dict`. Hot paths use `asyncio.gather`. No `asyncio.run()` inside a running loop.
- **Logging:** `structlog.get_logger(__name__)`. Pass kwargs as structured fields, not f-strings.
- **Tests:** `tests/` mirrors source layout. Unit tests mock LLM/Qdrant; integration tests live under `tests/test_integration/`. Use `@pytest.mark.integration` for slow ones.

---

## 10. Where to look first when something breaks

| Symptom | First file to read |
|---|---|
| RBAC leak / cross-tenant docs | `retrieval/hybrid_search.py` (search method, lines around `allowed_doc_ids`) |
| Hallucinated citations | `core/agents/faithfulness.py`, `core/agents/synthesizer.py::_extract_citations` |
| Streaming hangs / drops | `core/graph.py::run_rag_pipeline_stream` and the synth writer dispatch |
| Postgres checkpointer fails | `core/graph.py::_try_async_postgres_saver` (Windows event-loop pin) |
| Streamlit blank or `OSError [Errno 22]` | `utils/logging.py` (bootstrap at import) |
| Audit chain breaks | `utils/audit.py::compute_hash`, `scripts/verify_audit_chain.py` |
| Cloud router never used | `inference/router.py`, `query_sensitivity` in state |
| **Groq 429 on every BYOK chat** | `Dockerfile.hf` ADR-030 env vars + `inference/router.py::_get_model_for_provider` + `core/agents/evaluator.py` skip flag + `core/agents/retriever.py` grader-bypass branch |
| **Upload returns non-JSON** (Vercel Edge 30 s timeout) | `secureagentrag-web/src/lib/uploads.ts` text-then-parse fallback |
| **Sensitivity disclaimer in BYOK response** | `core/agents/synthesizer.py::_build_system_prompt` byok-mode gate + `_add_disclaimers` early-return |
| **Per-IP throttle bucket shared across visitors** | `interfaces/byok.py::client_ip_from_request` must read `X-Forwarded-For` leftmost token (HF reverse proxy masks `request.client.host`) |
| **BYOK upload 6 MB rejection unclear** | `interfaces/api.py /byok/uploads` raises 413 with chunk + size context; frontend maps to actionable copy |

---

## 11. If you are an AI agent reading this

Read `AGENTS.md` next. It is the operating manual.
