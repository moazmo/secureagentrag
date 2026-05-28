# Configuration Reference

Every runtime knob in SecureAgentRAG is a [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) field on `Settings` in [`config/settings.py`](../config/settings.py). All of them are read from the environment with the **`SAR_`** prefix (case-insensitive) or from a `.env` file in the working directory.

> **Canonical names.** The environment variable for a field is always `SAR_` + the field name upper-cased. For example the field `byok_owner_key_quota_per_hour` is set by `SAR_BYOK_OWNER_KEY_QUOTA_PER_HOUR`. Because `Settings` uses `extra="ignore"`, an unknown variable (a typo, or an old name) is **silently dropped** — it does not raise, it just has no effect. If a setting "won't take", check the name against this page first.

Sections below mirror the order of `config/settings.py`. Only non-secret defaults are shown; secrets (`*_API_KEY`, `SAR_JWT_SECRET`, `SAR_QDRANT_API_KEY`) default to unset and must be supplied via the environment or a secrets panel, never committed.

---

## Application

| Variable | Default | Purpose |
|---|---|---|
| `SAR_APP_NAME` | `SecureAgentRAG` | Display name. |
| `SAR_DEBUG` | `false` | Pretty console logs + verbose tracebacks. |
| `SAR_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR`. |

## Qdrant vector store

| Variable | Default | Purpose |
|---|---|---|
| `SAR_QDRANT_URL` | `http://localhost:6333` | REST endpoint. Qdrant Cloud uses `https://<uuid>.<region>.aws.cloud.qdrant.io`. |
| `SAR_QDRANT_COLLECTION` | `documents` | Base collection name. |
| `SAR_QDRANT_API_KEY` | _(unset)_ | Required for Qdrant Cloud; unset for local. |

## Ollama / LLM

| Variable | Default | Purpose |
|---|---|---|
| `SAR_OLLAMA_URL` | `http://localhost:11434` | Local inference + embedding server. |
| `SAR_LLM_MODEL` | `qwen3:8b` | Default generation model. |
| `SAR_EMBEDDING_MODEL` | `bge-m3` | Embedding model (1024-dim, multilingual). |
| `SAR_EMBEDDING_DIM` | `1024` | Must match the embedding model. |
| `SAR_EMBEDDING_BACKEND` | `ollama` | `ollama` or `local` (sentence-transformers). |
| `SAR_LOCAL_EMBEDDING_MODEL` | `BAAI/bge-m3` | Used when backend is `local`. |
| `SAR_OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps a model resident in VRAM. |

## Chunking

| Variable | Default | Purpose |
|---|---|---|
| `SAR_CHUNK_SIZE` | `1000` | Characters per chunk. |
| `SAR_CHUNK_OVERLAP` | `200` | Overlap between adjacent chunks. |

## Retrieval

| Variable | Default | Purpose |
|---|---|---|
| `SAR_TOP_K` | `10` | Candidates fetched before grading. |
| `SAR_RERANK_TOP_K` | `5` | Candidates kept after reranking. |
| `SAR_RELEVANCE_THRESHOLD` | `0.7` | Min relevance ratio before the corrective loop rewrites the query. |
| `SAR_RAG_FUSION_ENABLED` | `true` | Generate query reformulations and RRF-fuse them. Costs N-1 extra LLM calls. |
| `SAR_RAG_FUSION_N_QUERIES` | `3` | Number of reformulations when fusion is on. |

## Reranker

| Variable | Default | Purpose |
|---|---|---|
| `SAR_RERANKER_TYPE` | `none` | `none` / `cross_encoder` / `colbert` / `fine_tuned`. |
| `SAR_RERANKER_CHECKPOINT` | `BAAI/bge-reranker-v2-m3` | Cross-encoder checkpoint (~600 MB on first use). |
| `SAR_COLBERT_CHECKPOINT` | `colbert-ir/colbertv2.0` | ColBERT checkpoint (~400 MB). |
| `SAR_FINETUNED_RERANKER_PATH` | `data/checkpoints/reranker-domain-v1` | Local checkpoint for `fine_tuned`. |

## Inference providers

| Variable | Default | Purpose |
|---|---|---|
| `SAR_DEFAULT_PROVIDER` | `ollama` | Provider used when no override / sensitivity rule applies. |
| `SAR_CLOUD_PROVIDER` | _(unset)_ | Preferred cloud provider (`groq` / `openai` / `anthropic`). |
| `SAR_GROQ_API_KEY` / `SAR_OPENAI_API_KEY` / `SAR_ANTHROPIC_API_KEY` | _(unset)_ | Cloud credentials. |
| `SAR_GROQ_MODEL` | `llama-3.1-8b-instant` | Pinned so the demo doesn't default-drift to 70b. |
| `SAR_OPENAI_MODEL` | `gpt-4o-mini` | OpenAI default. |
| `SAR_ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Anthropic default. |
| `SAR_GROQ_API_BASE` / `SAR_OPENAI_API_BASE` / `SAR_ANTHROPIC_API_BASE` | provider URLs | Override for proxies / compatible gateways. |

## Pipeline thresholds

| Variable | Default | Purpose |
|---|---|---|
| `SAR_RELEVANCE_RETRY_THRESHOLD` | `0.5` | Below this, the rewriter fires. |
| `SAR_CONFIDENCE_THRESHOLD` | `0.6` | Below this, the evaluator flags `needs_human_review`. May be overridden by `evaluation/calibration.json`. |
| `SAR_MAX_RETRIES` | `2` | Max corrective rewrite cycles. |
| `SAR_REQUEST_TIMEOUT_S` | `60` | Hard wall-clock SLO budget for one pipeline run. `0` disables. |

## Embedding batching

| Variable | Default | Purpose |
|---|---|---|
| `SAR_EMBEDDING_BATCH_SIZE` | `32` | Texts per embedding call. |
| `SAR_EMBEDDING_MAX_CONCURRENT_BATCHES` | `4` | Parallel embedding batches. |

## Authentication (JWT)

| Variable | Default | Purpose |
|---|---|---|
| `SAR_JWT_SECRET` | _(unset)_ | HS256 signing secret. **When unset, the FastAPI/MCP layers fall back to an unsigned base64 dev token and warn on every request — never ship to prod without this** (or RS256). |
| `SAR_JWT_ALGORITHM` | `HS256` | `HS256` (HMAC) or `RS256` (public-key / JWKS). |
| `SAR_JWT_ISSUER` | `secureagentrag` | Checked against `iss` when present. |
| `SAR_JWT_AUDIENCE` | `secureagentrag-api` | Checked against `aud` when present. |
| `SAR_JWT_TTL_SECONDS` | `3600` | Lifetime of tokens minted by the dev `/token` endpoint. |
| `SAR_JWKS_URL` | _(unset)_ | IdP JWKS endpoint, used when algorithm is `RS256`. |
| `SAR_JWKS_CACHE_TTL_SECONDS` | `300` | TTL for cached JWKS public keys. |

## Faithfulness gate (NLI)

| Variable | Default | Purpose |
|---|---|---|
| `SAR_FAITHFULNESS_GATE_ENABLED` | `false` | Per-sentence entailment check after synthesis. One LLM call per cited sentence. |
| `SAR_FAITHFULNESS_GATE_MODE` | `flag` | `flag` (annotate `*[unsupported]*`) or `drop` (remove the sentence). |
| `SAR_FAITHFULNESS_THRESHOLD` | `0.7` | Min entailment ratio for a sentence to count as supported. May be set by calibration. |
| `SAR_FAITHFULNESS_MAX_CONCURRENT` | `4` | Parallel NLI checks. |

## Guardrails

| Variable | Default | Purpose |
|---|---|---|
| `SAR_GUARDRAILS_ENABLED` | `true` | Regex prompt-injection gate before retrieval. |
| `SAR_GUARDRAILS_STRICT` | `false` | After the regex gate, escalate to an LLM classifier. |
| `SAR_GUARDRAILS_BACKEND` | `llm` | `llm` (legacy SAFE/UNSAFE) or `llamaguard` (Meta S1–S14 taxonomy). |
| `SAR_LLAMAGUARD_MODEL` | `llama-guard3:8b` | Ollama tag for the LlamaGuard backend. |

## RBAC, multi-tenancy & PII

| Variable | Default | Purpose |
|---|---|---|
| `SAR_ENABLE_RBAC` | `true` | Enforce Qdrant payload-filter RBAC. |
| `SAR_MULTI_TENANT_COLLECTIONS` | `false` | Per-org collections (`documents_<org_id>`) instead of payload-level isolation only. |
| `SAR_PII_REDACTION_ENABLED` | `true` | Scrub PII + API keys before audit / cache persistence. |

## Sparse vectors

| Variable | Default | Purpose |
|---|---|---|
| `SAR_SPARSE_BACKEND` | `bm25` | `bm25` (zero deps) or `splade` (needs `[embeddings-local]`). |
| `SAR_SPARSE_VECTOR_NAME` | `sparse` | Named sparse vector inside the Qdrant collection. |
| `SAR_SPARSE_MODEL` | `naver/splade-cocondenser-ensembledistil` | Used when backend is `splade`. |

## Advanced retrieval (opt-in)

| Variable | Default | Purpose |
|---|---|---|
| `SAR_CONTEXTUAL_RETRIEVAL_ENABLED` | `false` | Prepend an LLM-written context to each chunk before embedding (Anthropic technique). One cheap LLM call per chunk at ingest. |
| `SAR_HYDE_ENABLED` | `false` | Embed a hypothetical answer instead of the raw query. |
| `SAR_SELF_QUERY_ENABLED` | `false` | Extract structured metadata filters from the query and merge with the RBAC filter. |
| `SAR_VLM_OCR_ENABLED` | `false` | Use a vision-language model as the primary OCR path (falls back to PaddleOCR). |
| `SAR_VLM_OCR_MODEL` | `qwen2.5-vl` | VLM tag for OCR. |
| `SAR_MULTIMODAL_DESCRIPTIONS_ENABLED` | `false` | Generate a text description for ingested images and embed it as a chunk. |

## Storage & persistence

| Variable | Default | Purpose |
|---|---|---|
| `SAR_AUDIT_LOG_DIR` | `audit_logs` | Daily JSONL audit files. |
| `SAR_CONVERSATION_DIR` | `conversations` | Saved conversations. |
| `SAR_CHECKPOINT_DB_PATH` | `data/checkpoints.sqlite` | LangGraph SQLite checkpoint DB. |
| `SAR_USE_PERSISTENT_CHECKPOINTER` | `false` | Enable SQLite/Postgres checkpointing. Off by default so pytest-asyncio loops don't collide with aiosqlite. |
| `SAR_POSTGRES_URL` | `postgresql://sar_user:sar_password@localhost:5433/secureagentrag` | Postgres checkpoint backend. |
| `SAR_PHOENIX_ENDPOINT` | _(unset)_ | Arize Phoenix OTEL collector URL. |
| `SAR_REDIS_URL` | `redis://localhost:6379/0` | Redis for distributed rate limiting / caching. |
| `SAR_USE_REDIS_RATE_LIMITER` | `false` | Use the Redis-backed rate limiter instead of the in-memory one. |

## BYOK demo mode

> These only matter when `SAR_BYOK_MODE=true`. They change the meaning of the pipeline — read [ADR-025](../DECISIONS.md) and [ADR-030](../DECISIONS.md), and [`BYOK_PRIVACY_TRADEOFFS.md`](./BYOK_PRIVACY_TRADEOFFS.md), before flipping any of them.

| Variable | Default | Prod (HF Space) | Purpose |
|---|---|---|---|
| `SAR_BYOK_MODE` | `false` | `true` | Master gate: per-request keys, session collections, cost-cut toggles, Phoenix off. |
| `SAR_BYOK_OWNER_KEY_QUOTA_PER_HOUR` | `3` | `10` | Per-IP throttle on the owner key. Visitor BYOK keys bypass it. **Canonical name — `SAR_BYOK_OWNER_QUOTA` is _not_ read.** |
| `SAR_SESSION_COLLECTION_TTL_HOURS` | `24` | `24` | Auto-purge cutoff for `documents_sess_<sid>` collections. **Canonical name — `SAR_SESSION_TTL_HOURS` is _not_ read.** |
| `SAR_CORS_ALLOW_ORIGINS` | `[]` | Vercel allowlist | JSON array of allowed origins. Empty = no CORS middleware mounted. |
| `SAR_ALLOW_CLOUD_FOR_HIGH` | `false` | `true` | Allow HIGH-sensitivity content on cloud. **Breaks the "HIGH stays local" guarantee** — only `true` because the HF Space has no Ollama. See the privacy doc. |
| `SAR_BYOK_AUDIT_MAX_ENTRIES` | `50` | `50` | Cap on `/byok/audit` rows. |
| `SAR_BYOK_UPLOAD_MAX_BYTES` | `5242880` (5 MB) | same | Per-file upload cap. |
| `SAR_BYOK_UPLOAD_MAX_FILES` | `5` | same | Per-session file cap. |
| `SAR_BYOK_UPLOAD_MAX_CHUNKS_PER_FILE` | `60` | same | Reject chatty PDFs that would blow the SLO budget. |
| `SAR_BYOK_UPLOAD_ALLOWED_EXTENSIONS` | `[".txt",".md",".pdf"]` | same | Upload allowlist. |
| `SAR_BYOK_SKIP_GRADER` | `true` | `true` | Bypass the per-doc LLM grader (cost). Trust embedding + RRF ordering. |
| `SAR_BYOK_SKIP_EVALUATOR` | `true` | `true` | Bypass the evaluator's LLM calls; use heuristic confidence. |

## Pricing (cost dashboard)

`SAR_PRICE_*_PER_1M` (input/output, per provider) and `SAR_PRICE_LOCAL_PER_SECOND` feed `evaluation/cost.py` to convert recorded usage into `$/query`. Defaults track public list prices; see `config/settings.py` for current values.

---

## Calibration override

`evaluation/calibration.json` (produced by `scripts/calibrate_thresholds.py`) can override `confidence_threshold` and `faithfulness_threshold` at startup so deployments inherit tuned values automatically. An explicit `SAR_CONFIDENCE_THRESHOLD` / `SAR_FAITHFULNESS_THRESHOLD` env var still wins. See `_apply_calibration()` in `config/settings.py`.
