# CLAUDE.md — Project Brief for AI Coding Agents

This file is the canonical entry point for any AI agent (Claude / Hermes / Kimi / Cursor / Aider) picking up work on SecureAgentRAG. Read it before touching code.

> 🚀 **Active branch: `deploy/prod-launch`.** A production launch is in progress (P6 from `private/roadmap.md`). Read [`launch-plan/12-agent-handoff.md`](./launch-plan/12-agent-handoff.md) **before** continuing any work — it is the operating contract for this launch and overrides any conflicting instruction below. `main` is frozen at `56c8c98`; do not push to it directly.

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
| Observability | structlog + Arize Phoenix | `utils/logging.py`, `utils/observability.py` |
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
├── tests/                  # pytest, 497 passing
├── helm/secureagentrag/    # Kubernetes manifests
├── deploy/                 # docker-compose auth profile + keycloak-realm.json
├── data/agent_evidence/    # 24-scenario gate evidence (results.md, screenshots)
├── sample_docs/            # PDF + txt corpus (incl. real NIST AI RMF)
├── DECISIONS.md            # ADR-001..022
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

## 5. State of the codebase (as of `e848ecf` on `deploy/prod-launch`)

- **620 tests pass**, 0 failed. Lint + format clean. (Baseline lifted +136 from 484 during the BYOK launch series.)
- **~33.8k Python LOC** across 161 files.
- **30 ADRs** in `DECISIONS.md` (24 historical + ADR-025/026/027/028 promoted from launch drafts + ADR-029 BYOK uploads + ADR-030 Groq cost optimisations).
- **18 commits ahead of `main`** on `deploy/prod-launch`. `main` frozen at `56c8c98` until launch closes.
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
  - `56c8c98` *(frozen `main` HEAD)* feat(retrieval): cache per-tenant QdrantManagers + pin sparse isolation (ADR-024)

---

## 5.1 BYOK production mode (the live demo's runtime contract)

Lives behind `SAR_BYOK_MODE=true`. The HF Space Dockerfile sets it; local dev does not.

- **Request shape:** `X-Demo-Persona` (engineer / compliance / executive — preset RBAC), `X-Session-ID` (UUID, drives session collection), `X-User-LLM-Key` (optional — visitor BYOK unlock), `X-User-Provider` (groq / openai / anthropic), `X-User-Ollama-URL` (optional). Extracted in `interfaces/byok.py`.
- **Endpoints (under `/byok/`):** `chat` (sync JSON), `chat/stream` (SSE: open|phase|token|blocked|final|error), `audit` (last-N session-scoped rows for export), `uploads` GET/POST/DELETE (5 MB · 5 files · 60 chunks/file · txt/md/pdf — see ADR-029).
- **Per-IP throttle:** `SAR_BYOK_OWNER_KEY_QUOTA_PER_HOUR=10` against owner key. Visitor BYOK bypasses. IP from `X-Forwarded-For` leftmost token (HF Spaces reverse proxy masks `request.client.host`).
- **Persona presets:** `_DEMO_PERSONAS` in `interfaces/api.py` maps each persona to `(clearance, roles, style)`. Style is threaded into the synth system prompt via `GraphState.persona_style`.
- **Cost-cut toggles (ADR-030):** `SAR_GROQ_MODEL=llama-3.1-8b-instant`, `SAR_RAG_FUSION_ENABLED=false`, `SAR_BYOK_SKIP_EVALUATOR=true`, `SAR_BYOK_SKIP_GRADER=true`, `SAR_FAITHFULNESS_GATE_ENABLED=false`, `SAR_RERANKER_TYPE=none`, `SAR_RELEVANCE_THRESHOLD=0.55`, `SAR_MAX_RETRIES=1`, `SAR_RERANK_TOP_K=10`. Router classifier short-circuits queries ≤80 chars to `query_type="simple"`. Net effect: ~2 Groq calls/chat vs ~5–6 before.
- **HIGH-on-cloud unlock:** `SAR_ALLOW_CLOUD_FOR_HIGH=true` in production because the HF Space has no Ollama. Frontend renders a `sensitivity:` badge so the visitor is informed. *Hero claim "HIGH never leaves local" is true in self-hosted mode only.*
- **Session collections:** `documents_sess_<sanitized_session_id>`. Dual-collection retrieval (base ∪ session) under one RBAC filter; RRF-fused. 24 h TTL via `SAR_SESSION_TTL_HOURS=24` + `scripts/byok_session_purge.py`.
- **Audit:** session-scoped only (`/byok/audit` filters `user_id == "demo-<sid>"`). SHA-256 chain intact; downloadable JSONL.
- **Sensitivity disclaimer suppressed** in BYOK mode (both prompt-side gate in `_build_system_prompt` and post-synth `_add_disclaimers` early-return). The frontend's `sensitivity:` badge is the user-facing signal.
- **No Phoenix / Postgres / Ollama in BYOK mode.** Audit on /tmp; checkpointer in-memory.

## 6. Genuinely remaining work (audited 2026-05-27)

The 80-task BYOK launch is complete. Pipeline survives Egypt-from-mobile traffic on $0/mo. What's left:

- **Phase 8 — record 4-minute demo video** against `secureagentrag-web.vercel.app`. Owner action. Script lives in `launch-plan/08-demo-video.md`.
- **Phase 9 — merge `deploy/prod-launch` → `main`, tag `v1.0.0-launch`**. After Phase 8.
- **Optional:** upload fine-tuned reranker to `LeomordKaly/secureagentrag-reranker-v1` HF Hub model repo and flip `SAR_RERANKER_TYPE=fine_tuned` on the Space. Skipped for now — 10-doc corpus does not benefit materially.
- **Optional:** selective guardrails escalation (regex hit → LlamaGuard on suspicious only). Would catch unicode-obfuscation without burning Groq budget on every chat.

**Recently shipped** (was in this section, now done):
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
