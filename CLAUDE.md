# CLAUDE.md — Project Brief for AI Coding Agents

This file is the canonical entry point for any AI agent (Claude / Hermes / Kimi / Cursor / Aider) picking up work on SecureAgentRAG. Read it before touching code.

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

## 5. State of the codebase (as of this commit)

- **497 tests pass**, 0 skipped. Lint + format clean.
- **~28.5k Python LOC** across 138 files.
- **9 graph nodes:** router → guardrails → security → retriever → grader → rewriter → synthesizer → faithfulness → evaluator.
- Streamlit, FastAPI, MCP all share `core.schemas.QueryResponse`.
- **Hybrid search** uses Qdrant native sparse vectors (BM25 or SPLADE backend). The legacy `rank_bm25` pickle, `utils/file_lock.py`, and the post-fusion RBAC re-check are all gone. Sparse runs under the same RBAC filter as dense — cross-tenant bypass is structurally impossible.
- **Auth** dispatches HS256 (default, dev) ↔ RS256 (Keycloak/Auth0 via JWKS) on `SAR_JWT_ALGORITHM`.
- **Guardrails** escalation routes through `regex` → `llm` → `llamaguard` per `SAR_GUARDRAILS_BACKEND`. LlamaGuard 3 maps S1-S14 categories to audit-friendly reasons.
- **Reranker** factory accepts `none` / `cross_encoder` / `colbert` / `fine_tuned`. Training + bench scripts in tree; actual fine-tune is opt-in GPU work.
- **24-scenario UI gate (H.1 + H.2) is 24/24 PASS** on this HEAD, evidence under `data/agent_evidence/`.
- Recent commits (chronological, newest first):
  - `1bcde26` test(h2-gate): 12/12 advanced real-world scenarios PASS
  - `2f0e28d` feat(retrieval): fine-tuned reranker scaffolding (P4)
  - `45ebfde` refactor(inference): consolidate Groq + OpenAI clients via shared parent (-203 LOC)
  - `6722772` refactor(ui): extract chat service helpers into app/chat_service.py (-162 LOC in view)
  - `038fdae` feat(guardrails): LlamaGuard 3 as drop-in escalation backend (P3)
  - `5cda492` docs(evidence): 24-scenario UI gate H.1 PASS (H.2 was deferred at the time)
  - `1a22ab8` docs: README + CLAUDE.md + architecture + ADR-019/020 (SPLADE + RS256)
  - `b43bc66` feat(ops): Keycloak under docker-compose auth profile + realm export
  - `e80a519` feat(auth): RS256 + JWKS verification path with in-memory key cache
  - `26500ae` feat(retrieval): Qdrant native sparse vectors replace rank_bm25 pickle

---

## 6. Genuinely remaining work (audited 2026-05-21)

In priority order. Each one is its own commit/PR-worthy unit.

1. **Actually run the fine-tune** — P4 scaffolding is in tree (`scripts/train_reranker.py` + `scripts/bench_reranker.py`). The 1-2 GPU-hour training run is left to whoever owns the GPU box. Output: `data/checkpoints/reranker-domain-v1/`. Bench against baseline must show ≥1pp NDCG@10 lift.
2. **Calibrate confidence + faithfulness thresholds** against a labeled gold set (Ragas). Wire into CI so >5pp regression fails the build. Needs the gold set to exist first.
3. **Deeper `app/views/chat.py` slim** — currently 621 LOC. The streaming state machine, thread sidebar, and cached-render helpers could extract next to land it under 300.
4. **Per-tenant SPLADE indexes** — when `SAR_MULTI_TENANT_COLLECTIONS=true`, each org should get its own sparse index slot in its Qdrant collection. Today the SPLADE vector field is shared across the multi-tenant boundary.
5. **`evaluation/nist_rerank_gold.jsonl`** — hand-labelled 20-query subset for `scripts/bench_reranker.py`'s in-domain bench. Not present yet; the script gracefully skips when absent.

**Recently shipped** (was in this section, now done):
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

---

## 11. If you are an AI agent reading this

Read `AGENTS.md` next. It is the operating manual.
