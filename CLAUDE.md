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
| Sparse | rank_bm25 pickle (per-process) | `retrieval/hybrid_search.py` |
| Rerankers | BGE-Reranker-v2-M3 / ColBERTv2 (optional) | `retrieval/reranker.py`, `retrieval/colbert_reranker.py` |
| Cloud LLMs | Groq / OpenAI / Anthropic via httpx | `inference/cloud_clients.py` |
| Persistence | AsyncPostgres → AsyncSqlite → MemorySaver | `core/graph.py::_get_*_checkpointer` |
| UI | Streamlit (sidebar + 4 tabs) | `app/main.py`, `app/views/*` |
| API | FastAPI (`:8080`) + MCP stdio | `interfaces/api.py`, `interfaces/mcp_server.py` |
| Auth | HS256 JWT via python-jose | `utils/auth.py` |
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
│   ├── qdrant_client.py    # RBAC filter, multi-tenant for_org
│   ├── hybrid_search.py    # dense+BM25+RRF, RBAC re-check on BM25
│   ├── embeddings.py
│   ├── reranker.py / colbert_reranker.py
│   ├── self_query.py       # NL → structured filter
│   ├── hyde.py
│   └── multitenancy.py     # get_collection_name
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
├── evaluation/             # ragas, cost dashboard, benchmark, nightly
├── utils/                  # auth, audit, pii, logging, rate_limiter, ...
├── scripts/                # smoke, seed, interview_demo, bench
├── tests/                  # pytest, 484 passing
├── helm/secureagentrag/    # Kubernetes manifests
├── sample_docs/            # PDF + txt corpus (incl. real NIST AI RMF)
├── DECISIONS.md            # ADR-001..018
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

- **484 tests pass**, 3 skipped. Lint + format clean.
- **~26.7k Python LOC** across 130 files.
- **9 graph nodes:** router → guardrails → security → retriever → grader → rewriter → synthesizer → faithfulness → evaluator.
- Streamlit, FastAPI, MCP all share `core.schemas.QueryResponse`.
- Recent commits (chronological):
  - `b8cdaa2` fix(ui): role multiselect expanded to all seeded roles
  - `3b2cab7` fix(security): close BM25-bypass RBAC leak when dense returns zero
  - `807636a` docs(readme): correct stale Planned section — 8 items already shipped
  - `2164a81` docs: delete REVIEW_REPORT.md + refresh architecture.md
  - `9b8dc19` chore(scripts): merge cloud_bench_quick + seed_demo_rbac via flags
  - `79f2af8` fix(logging): bootstrap structlog at import time (Streamlit Windows)
  - `9c39229` refactor(streaming): unify via graph.astream(stream_mode=updates+custom)
  - `069c3e5` feat(demo): interview_demo + README tighten
  - `025ce73` feat(auth): HS256 JWT replacing unsigned base64
  - `a75dd33` feat(faithfulness): NLI gate + evaluator integration
  - `23c7ce6` feat(slo): SAR_REQUEST_TIMEOUT_S deadline
  - `40b2cf9` fix(synthesizer): refuse when grader rejects all docs

---

## 6. Genuinely remaining work (the "planned" list, audited 2026-05-20)

In priority order. Each one is its own commit/PR-worthy unit.

1. **SPLADE / Qdrant native sparse vectors** — retire `rank_bm25` pickle. Per-tenant by collection, no cross-tenant leakage class. Touches `retrieval/hybrid_search.py`, `retrieval/embeddings.py`, `ingestion/pipeline.py`. Likely **-300 LOC** when done. Risk: medium (requires re-indexing).
2. **LlamaGuard / NeMo Guardrails classifier** — drop-in classifier as an alternative to the local-LLM escalation path in `core/agents/guardrails_llm.py`. Touches that file + new settings flag. Risk: low.
3. **Fine-tuned domain reranker** — train a cross-encoder on labeled query/passage pairs from MS-MARCO or in-domain data. Touches `retrieval/reranker.py` + new `scripts/train_reranker.py`. Risk: low (off-the-shelf fallback always works).
4. **RS256 + JWKS auth** — swap HS256 for IdP-driven public-key verification in `utils/auth.py::_verify_jwt`. The hook is 30 LOC; swap is ~20 LOC. Add `docker compose` profile with Keycloak. Touches `utils/auth.py`, `docker-compose.yml`. Risk: low.

Plus continuous improvement of the main goal:
- **Slim `app/views/chat.py`** (currently 783 LOC) — extract a `chat_service.py` so the view is UI-only. Less code, easier review.
- **Calibrate confidence + faithfulness thresholds** against a labeled gold set (Ragas).
- **Per-tenant BM25 / SPLADE indexes** when (1) lands.

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
