# Architecture Decision Records (ADR)

This document captures key architectural decisions for SecureAgentRAG. Each ADR explains the context (problem), decision (what we chose), and consequences (tradeoffs).

---

## ADR-001: uv over Poetry for Package Management

**Date:** 2025-01-10  
**Status:** Accepted

**Context:**
We need a fast, reliable Python package manager that supports modern standards (PEP 621/631) and provides reproducible builds. Poetry is mature but slow on dependency resolution. pip-tools lacks virtual environment management. The project has many heavy dependencies (PyTorch, PaddleOCR, LangGraph) that stress resolver performance.

**Decision:**
Use [uv](https://github.com/astral-sh/uv) as the sole package manager with `pyproject.toml` (PEP 621).

**Consequences:**
- (+) 10-100x faster dependency resolution compared to Poetry/pip
- (+) Native support for `pyproject.toml` (PEP 621) — no proprietary config format
- (+) Built-in virtual environment management (`uv sync` creates and populates venv)
- (+) Rust-based — no Python bootstrapping issues on fresh systems
- (+) Deterministic lockfile for reproducible deployments
- (-) Younger ecosystem; some teams may be unfamiliar with uv
- (-) Lockfile format differs from Poetry's `poetry.lock`
- (-) Fewer IDE integrations compared to Poetry (rapidly improving)

---

## ADR-002: Qdrant over Chroma/Pinecone for Vector Storage

**Date:** 2025-01-10  
**Status:** Accepted

**Context:**
We need a vector database that supports metadata filtering (for RBAC), scales from development to production, and provides payload-based access control without application-level workarounds. Chroma is embedded-only and lacks advanced filtering. Pinecone is cloud-hosted (conflicts with privacy-first requirement).

**Decision:**
Use Qdrant as the vector store with Docker deployment. Enforce RBAC through Qdrant's native payload filtering on every retrieval query.

**Consequences:**
- (+) Native payload filtering enables RBAC at the DB level — filters applied before results are returned
- (+) Production-grade with replication, sharding, and quantization support
- (+) gRPC and REST APIs with official Python client (`qdrant-client`)
- (+) Supports multiple distance metrics (cosine, dot, euclidean)
- (+) Self-hosted — data never leaves local infrastructure
- (+) Quantization support (scalar, product) for memory optimization
- (-) Requires a separate service (Docker container) — more operational overhead than embedded Chroma
- (-) Slightly more complex setup for development environments
- (-) No managed cloud offering needed (self-hosted is intentional for privacy)

---

## ADR-003: LangGraph over Plain LangChain Agents for Orchestration

**Date:** 2025-01-12  
**Status:** Accepted

**Context:**
The corrective RAG pattern requires conditional branching, loops (retry on low relevance), and stateful multi-step workflows. Plain LangChain chains are linear; LCEL helps but doesn't support cycles natively. LangChain agents use tool-calling which doesn't match our fixed pipeline structure.

**Decision:**
Use LangGraph for the multi-agent orchestration layer with a `StateGraph` and `TypedDict` state.

**Consequences:**
- (+) First-class support for cycles (grader → rewriter → retriever loop)
- (+) Conditional edges for branching logic (security gate, relevance threshold)
- (+) Built-in state management with TypedDict — type-safe and IDE-friendly
- (+) MemorySaver checkpointer for conversation persistence
- (+) Visual graph debugging and LangSmith integration
- (+) Each node is a pure function — easy to test in isolation
- (-) Additional abstraction layer to learn (graph mental model)
- (-) Tighter coupling to LangChain ecosystem for some utilities
- (-) Requires careful state schema design upfront

---

## ADR-004: Qwen3-8B as Default Local LLM

**Date:** 2025-01-15  
**Status:** Accepted

**Context:**
We need a local LLM that: (1) fits in 8GB VRAM when quantized, (2) supports multilingual text including Arabic, (3) provides strong reasoning and instruction-following, and (4) runs efficiently via Ollama. Candidates: Llama-3.1-8B, Mistral-7B, Qwen2.5-7B, Qwen3-8B.

**Decision:**
Use Qwen3-8B (Q4_K_M quantization, ~5.5GB VRAM) as the default generation model via Ollama.

**Consequences:**
- (+) Excellent multilingual performance — top-tier Arabic, English, Chinese support
- (+) Strong reasoning and instruction-following at 8B parameter scale
- (+) Fits comfortably in 8GB VRAM with Q4_K_M quantization (~5.5GB)
- (+) Apache 2.0 license — no commercial restrictions
- (+) Active community with Ollama-native quantized builds
- (+) Supports thinking/reasoning mode for complex queries
- (-) Slightly larger than 7B alternatives — tighter VRAM budget when running alongside embeddings
- (-) Requires Ollama runtime (not a disadvantage for this architecture)
- (-) Newer model — less community tooling compared to Llama family

---

## ADR-005: RBAC Enforcement at Vector DB Level

**Date:** 2025-01-15  
**Status:** Accepted

**Context:**
Role-based access control can be implemented at multiple layers: application code (filter after retrieval), middleware (intercept requests), or database queries (filter during retrieval). Application-level filtering risks data leakage through bugs — the model could see unauthorized context even if the UI hides it. We want defense-in-depth.

**Decision:**
Enforce RBAC by storing access roles in Qdrant point payloads and applying `must` metadata filters on every retrieval query. Documents are tagged with allowed roles at ingestion time. Unauthorized documents are never retrieved, never embedded in prompts, and never seen by the LLM.

**Consequences:**
- (+) Access control enforced at query time — impossible to bypass via application bugs
- (+) No separate authorization service needed initially
- (+) Filtering happens in the DB engine — performant even at scale
- (+) LLM never sees unauthorized content (prevents indirect leakage through generation)
- (+) Audit trail shows exactly which documents were accessible
- (-) Role changes require re-indexing affected documents (metadata update)
- (-) Complex role hierarchies may need application-level resolution before filter construction
- (-) Cannot retroactively revoke access to already-generated cached responses

---

## ADR-006: No FastAPI Initially — Streamlit-Only Interface

**Date:** 2025-01-18  
**Status:** Accepted

**Context:**
This is a portfolio/demonstration project targeting interactive document Q&A. Adding a REST API layer (FastAPI) would increase complexity, require separate authentication, and duplicate logic without immediate benefit. The primary use case is interactive exploration through a web UI.

**Decision:**
Start with Streamlit as the sole user interface. Defer FastAPI to a future milestone if programmatic API access becomes necessary.

**Consequences:**
- (+) Faster development — single interface to maintain
- (+) Rich interactive UI with minimal code (file upload, chat, admin panels, charts)
- (+) Lower cognitive load for reviewers evaluating the project
- (+) Streamlit's caching and session state handles most needs
- (+) Hot reload during development for rapid iteration
- (-) No programmatic API access for external integrations
- (-) Streamlit's execution model (top-to-bottom rerun) requires careful state management
- (-) Will need FastAPI later if the platform serves other services or mobile clients
- (-) Limited to single-user sessions without additional session management

---

## ADR-007: Self-Implemented Chunker over LangChain Text Splitters

**Date:** 2025-01-20  
**Status:** Accepted

**Context:**
Text chunking is a critical pipeline stage that affects retrieval quality. LangChain provides `RecursiveCharacterTextSplitter` and other splitters, but they add a heavy dependency chain and don't handle our specific needs (page-aware chunking, metadata propagation, OCR text handling). We want full control over chunking behavior.

**Decision:**
Implement a custom `TextChunker` class with configurable chunk size, overlap, and page-aware splitting. No LangChain dependency for text processing.

**Consequences:**
- (+) Zero LangChain dependency for ingestion — lighter, faster imports
- (+) Full control over page boundary handling and metadata propagation
- (+) Can optimize for Arabic text (different sentence boundaries)
- (+) Easier to add semantic chunking later without framework constraints
- (+) Simpler debugging — no framework abstractions to trace through
- (-) Must maintain our own chunking logic (more code to test)
- (-) Missing LangChain community's battle-tested edge case handling
- (-) Need to implement overlap logic manually

---

## ADR-008: Hybrid Search with Reciprocal Rank Fusion

**Date:** 2025-01-22  
**Status:** Accepted

**Context:**
Pure dense retrieval (embeddings) excels at semantic matching but can miss exact keyword matches important in technical/legal documents. Pure sparse retrieval (BM25) handles keywords well but misses paraphrases. We need both for comprehensive recall, especially for bilingual documents where keyword matching supplements embedding quality.

**Decision:**
Combine dense retrieval (Qdrant + BGE-M3) with sparse retrieval (BM25) using Reciprocal Rank Fusion (RRF) with k=60 to merge result rankings.

**Consequences:**
- (+) Best-of-both-worlds: semantic understanding + keyword precision
- (+) RRF is robust and parameter-free (single k constant)
- (+) Cross-encoder reranker after fusion ensures final result quality
- (+) BM25 handles exact entity names, codes, and identifiers that embeddings may abstract away
- (+) Graceful degradation — if BM25 index unavailable, falls back to dense-only
- (-) Requires maintaining an in-memory BM25 index alongside the vector store
- (-) BM25 index doesn't benefit from Qdrant's RBAC filters (applied to dense only, fusion handles overlap)
- (-) Additional latency from running two search paths (mitigated by async execution)

---

## ADR-009: Conditional Imports for Optional Dependencies

**Date:** 2025-01-25  
**Status:** Accepted

**Context:**
The project depends on several heavy optional packages: PaddleOCR (~2GB with PaddlePaddle), Ragas (requires datasets + transformers), Arize Phoenix (OpenTelemetry stack), and sentence-transformers. Not all users need all features. CI/CD environments may want to run tests without installing GPU-dependent packages.

**Decision:**
Use `try/except ImportError` patterns for all optional dependencies. Core functionality (ingestion, retrieval, inference) works without optional packages. Each module defines an `_AVAILABLE` flag and gracefully degrades.

**Consequences:**
- (+) Core application runs with minimal dependencies
- (+) CI/CD can test business logic without installing heavy ML packages
- (+) Users can progressively enable features as needed
- (+) Import errors are caught and logged, never crash the application
- (+) Docker image can be layered (base + optional feature layers)
- (-) Must test both code paths (with and without optional deps)
- (-) Feature availability must be checked at runtime, not compile time
- (-) IDE autocomplete may not work for conditionally-imported modules

---

## ADR-010: Sensitivity-Based Inference Routing

**Date:** 2025-01-28  
**Status:** Accepted

**Context:**
The platform processes documents with varying sensitivity levels. Sending CONFIDENTIAL or RESTRICTED data to cloud LLM providers (OpenAI, Groq, Anthropic) violates privacy requirements. However, cloud providers offer better performance for non-sensitive workloads. We need a routing mechanism that respects data classification automatically.

**Decision:**
Implement an `InferenceRouter` that inspects the sensitivity level of retrieved documents and routes inference requests accordingly:
- HIGH → Local (Ollama) only, forced
- MEDIUM → Local by default, cloud if explicitly preferred and authorized
- LOW → Any configured provider based on preference

**Consequences:**
- (+) Privacy enforcement is automatic — developers don't need to remember sensitivity rules
- (+) Performance optimization — low-sensitivity queries can use faster/larger cloud models
- (+) Clear audit trail showing routing decisions and their rationale
- (+) Admin override available for exceptional cases
- (+) Graceful fallback — if cloud is unavailable, everything runs locally
- (-) Latency difference between local and cloud may confuse users
- (-) Sensitivity classification must be accurate (garbage in → wrong routing)
- (-) Cloud API keys stored in environment — requires secure secret management
- (-) Cost implications when cloud providers are enabled (usage-based billing)

---

## ADR-011: Tamper-Evident Audit Log (SHA-256 Hash Chain)

**Date:** 2026-05-19
**Status:** Accepted

**Context:**
Audit JSONL files were append-only but trivially mutable — anyone with disk
access could edit a prior entry to whitewash a security incident with no
record. For an "enterprise governance" pitch this is a gap. Three options
considered: signed entries (per-entry HMAC), Merkle tree, and hash chain.

**Decision:**
Each audit entry stores ``prev_hash`` (SHA-256 of the prior entry's hash)
and ``entry_hash`` (SHA-256 of its own canonical JSON minus the hash field
itself). A ``verify_chain()`` method and ``scripts/verify_audit_chain.py``
CLI walk the chain and detect any modification, insertion, or deletion.

**Consequences:**
- (+) Any in-place edit, re-ordering, or deletion is detected at verify time
- (+) Genesis-block style — first ever entry pins ``prev_hash="GENESIS"``
- (+) Verification works offline; no external timestamping service needed
- (+) Writers are serialised by an in-process RLock so concurrent appends
  cannot collide on the same ``prev_hash``
- (-) Multi-process writers must coordinate through a single writer or
  migrate to a transactional store (Postgres)
- (-) Removing an entry retroactively requires rewriting all subsequent
  hashes — by design

---

## ADR-012: Prompt-Injection Guardrails Node

**Date:** 2026-05-19
**Status:** Accepted

**Context:**
The security/RBAC node only validated that the user could access the
relevant document set. It did not catch attempts to override the system
prompt ("ignore previous instructions", "<system>...</system>",
``<|im_start|>system``) or extract internal state.

**Decision:**
Insert a new ``guardrails`` node between ``router`` and ``security``.
Regex-based pattern matching (9 classes: chat-template injection, system
tag, ignore/disregard instructions, prompt extraction, jailbreak persona,
role override, explicit bypass, privilege escalation, empty / overlong).
Blocks return early via a conditional edge — the request never spends
embedding or LLM budget.

**Consequences:**
- (+) Catches obvious prompt-injection without an extra LLM call
- (+) Blocked attempts are written to the audit log as ``security_block``
- (+) Order matters — patterns are listed most-specific first so high-
  signal markers (chat templates) win over broader matches
- (-) Regex misses semantic-only attacks (e.g. social-engineered context)
- (-) A future ``guardrails_strict`` mode could chain a Llama-Guard
  classifier on top — left as a hook

---

## ADR-013: Contextual Retrieval (Anthropic 2024)

**Date:** 2026-05-19
**Status:** Accepted (opt-in via ``SAR_CONTEXTUAL_RETRIEVAL_ENABLED``)

**Context:**
On long documents, BGE-M3 embeddings for short chunks are ambiguous
because the chunk loses its surrounding context. Anthropic reported a
35-49% reduction in retrieval failures by prepending a short LLM-generated
context summary to each chunk *before embedding*.

**Decision:**
Add an opt-in step in ``ingestion/pipeline.py``: for every chunk, call the
LLM with the full document + chunk and ask for a 1-3 sentence context. The
augmented text (``"Context: ...\n\nchunk"``) is what goes through embedding
and BM25 — the chunk text shown to users is unchanged. Generation is
parallelised via a bounded ``asyncio.Semaphore``.

**Consequences:**
- (+) Measurable recall gain on under-specified queries
- (+) BM25 + dense both benefit from the same augmented surface form
- (+) Display text unchanged — UX is unaffected
- (-) One LLM call per chunk at ingestion (cheap with local Qwen3-8B but
  not free on big corpora). Mitigated by parallelism and a per-prompt
  document truncation cap.
- (-) Storage doubles per chunk if both raw and augmented are kept (we
  store only the augmented version's embedding; raw text in payload)

---

## ADR-014: HyDE (Hypothetical Document Embeddings)

**Date:** 2026-05-19
**Status:** Accepted (opt-in via ``SAR_HYDE_ENABLED``)

**Context:**
Question-style queries (BGE-M3 input on "What are the four NIST AI RMF
functions?") sit in question-space, while the corpus passages sit in
statement-space. The dense vector for the query often does not align
neatly with the vector for the answering passage.

**Decision:**
When enabled, the retriever calls the LLM to write a short factual
passage that would answer the query, then concatenates that with the
original query and embeds *that*. The concatenation keeps BM25 happy
(original keywords still present) while the dense vector lands in
document-space.

**Consequences:**
- (+) Boosts recall on complex / vocabulary-mismatched queries
- (+) Gracefully falls back to the raw query on any LLM failure
- (-) Adds ~1-3s of LLM latency per query — skipped for ``simple`` /
  ``out_of_scope`` query types
- (-) The hypothetical answer can hallucinate; we only embed it, never
  show it, but garbage-in could still bias retrieval

---

## ADR-015: MCP + FastAPI Surfaces

**Date:** 2026-05-19
**Status:** Accepted

**Context:**
Streamlit alone could not satisfy two important consumption patterns:
(1) IDE-side agents (Claude Desktop, Claude Code, Cursor) want an MCP
endpoint; (2) programmatic / mobile / external services want a stable
REST contract.

**Decision:**
Add ``interfaces/mcp_server.py`` (FastMCP stdio transport) and
``interfaces/api.py`` (FastAPI). Both share the same ``QueryResponse``
Pydantic model defined in ``core/schemas.py`` so clients see one shape
regardless of transport. Auth on FastAPI is a stateless base64-bearer
that decodes to a ``UserContext`` — a hook left for Keycloak/Auth0 in
production.

**Consequences:**
- (+) IDE agents can call ``retrieve`` and ``query`` over MCP — the
  platform becomes an agent-callable tool
- (+) REST API enables external integration / mobile clients
- (+) Single Pydantic contract via ``QueryResponse.from_state(state)``
- (-) Two extra surfaces to test
- (-) Bearer-token decode is dev-grade; production must layer JWT
  validation in ``_resolve_user``

---

## ADR-016: PII Redaction Before Persistence

**Date:** 2026-05-19
**Status:** Accepted

**Context:**
The audit log and query cache previously stored raw query text and
document snippets. A query like "look up John Doe's SSN 123-45-6789"
would leak that PII into JSONL on disk and into Redis if enabled.

**Decision:**
A new ``utils/pii.py`` module scrubs strings just before they leave the
in-memory pipeline. Regex patterns cover email, phone, SSN, credit card
(Luhn-validated), IBAN, IPv4, URL-with-creds and API keys. If Microsoft
Presidio is installed (optional ``[pii]`` extra), it runs on top for
NER-based PII (names, locations, organisations).

**Consequences:**
- (+) Disk and Redis never see plaintext PII
- (+) Live in-flight state still has raw text so the model answers well —
  redaction is only at the storage boundary
- (+) The audit *hash chain* hashes the redacted JSON, so verification
  works on the redacted entry, not the pre-redacted one
- (-) Regex false-positives are possible (e.g. card-shaped phone numbers
  caught by Luhn). Mitigated by order: URL/email/SSN/CC/IBAN/IP/key first
- (-) Presidio is a heavy optional dep; not enabled by default

---

## ADR-017: Cost Model for Mixed Local / Cloud Inference

**Date:** 2026-05-19
**Status:** Accepted

**Context:**
The dashboard previously showed only latency — no view of $/query. With
cloud routing now first-class, ops needs to see (a) cloud spend by
provider and (b) the local-equivalent compute cost to make the privacy
trade-off legible.

**Decision:**
Per-1M-token prices for Groq / OpenAI / Anthropic live in
``config/settings.py``. Local inference is priced as
``latency_seconds × $/sec`` derived from typical consumer GPU draw
(200W × $0.15/kWh ≈ $0.000008/sec). ``evaluation/cost.py`` exposes
``estimate_query_cost`` consumed by the Streamlit dashboard.

**Consequences:**
- (+) Cost dashboard answers "what's cheaper per query, local or cloud?"
- (+) Prices are config — easy to adjust as provider prices change
- (-) Local cost is an estimate; real measurement would need power
  metering integration


---

## ADR-018: AsyncPostgresSaver for LangGraph Checkpointing

**Date:** 2026-05-19
**Status:** Accepted (opt-in via `SAR_USE_PERSISTENT_CHECKPOINTER`)

**Context:**
The pipeline previously cached a sync `PostgresSaver` opened via
`psycopg.Connection.connect`. Inside the async pipeline (`graph.ainvoke`)
this blocks the event loop on every write. The SQLite path already used
`AsyncSqliteSaver`, so the Postgres path was asymmetric. Worse: building
the sync saver inside `asyncio.run` clobbered langgraph internals on
Windows because psycopg async required the Selector loop, not the
default Proactor.

**Decision:**
1. Switch to `AsyncPostgresSaver` backed by an `AsyncConnectionPool`.
2. Pin `asyncio.WindowsSelectorEventLoopPolicy` at `core/graph.py` import
   time so subsequent `asyncio.run` invocations pick it up.
3. Add `build_rag_graph_async()` that awaits a fresh saver from inside
   the running loop; `run_rag_pipeline` calls it. Tests + sync callers
   keep using `build_rag_graph()` which still falls back to MemorySaver
   when nested inside a loop.
4. Map docker-compose Postgres to host port 5433 to avoid colliding with
   system-installed Postgres on the default 5432.

**Consequences:**
- (+) Postgres checkpoint reads/writes happen inside the same async loop
  the pipeline already uses — no thread bouncing.
- (+) Verified end-to-end: a single 113s pipeline run lands 9 checkpoint
  rows (one per node) in the `checkpoints` table.
- (+) Connection pool (1-5) handles concurrent FastAPI / Streamlit
  requests without saturating Postgres.
- (-) Persistence extras add `psycopg[binary,pool]` (~5 MB) on top of
  base install.
- (-) Windows users still need the selector loop policy — pinned at
  import time but documented in the RUNBOOK.

## ADR-019: HS256 + RS256 dispatch with JWKS-cached verification

**Status:** Accepted (2026-05-21)

**Context:**
The initial JWT layer (ADR-equivalent in [feat(auth): HS256-signed JWT
bearer tokens](https://github.com/moazmo/secureagentrag/commit/025ce73))
verified tokens with a shared HMAC secret. That secret has to live on
both the API server and every token issuer, which is fine for a single-
process demo but rules out external IdPs and key rotation across
services. Production deployments need public-key verification against
an identity provider's JWKS endpoint (Keycloak, Auth0, Microsoft Entra).

**Decision:**
1. Keep HS256 as the default (`SAR_JWT_ALGORITHM=HS256`) for local
   development and the existing test suite. Setting
   `SAR_JWT_ALGORITHM=RS256` plus `SAR_JWKS_URL=<idp jwks endpoint>`
   switches the verifier to public-key mode.
2. `utils/auth.py::_verify_jwt` dispatches on the algorithm flag. The
   RS256 branch reads the token's `kid` header, looks up the
   corresponding PEM in `utils/jwks_cache.py`, and hands it to
   `python-jose`'s `jwt.decode`. Missing `kid` → `AuthError("bad_claims")`.
   JWKS fetch errors → `AuthError("bad_signature")`.
3. `utils/jwks_cache.py` keeps a per-URL TTL cache (default 300s,
   `SAR_JWKS_CACHE_TTL_SECONDS`). On `kid` miss inside the TTL the
   cache refreshes once before giving up. On fetch error the cache
   serves stale keys if available — graceful degradation through a
   transient IdP outage.
4. Production deployments bring up Keycloak via the new compose profile
   (`docker compose --profile auth up -d keycloak`) which auto-imports
   the realm from `deploy/keycloak-realm.json`.

**Consequences:**
- (+) Public-key verification works against any standards-compliant
  OIDC provider — no code change needed past the JWKS URL.
- (+) Key rotation handled transparently by the cache: rotating keys
  in Keycloak triggers a JWKS refresh on the first request with the
  new `kid`. Old tokens remain valid until their `exp`.
- (+) Stable error-reason taxonomy (`missing` / `expired` /
  `bad_signature` / `bad_claims` / `malformed`) shared across HS256
  and RS256 so HTTP status code mapping stays unchanged.
- (-) Adds ~125 LOC for the cache + the dispatch logic. Justified by
  the production-readiness win.
- (-) `cryptography` is now required (transitively via `python-jose[cryptography]`)
  for the JWK→PEM conversion.

## ADR-020: Qdrant native sparse vectors over `rank_bm25` pickle

**Status:** Accepted (2026-05-21)

**Context:**
The hybrid retrieval path stored BM25 in a global `rank_bm25` pickle on
disk, behind a `FileLock` for concurrent writes. Two problems:

1. **RBAC.** BM25 has no payload filter, so the search ran on the full
   global index. The post-fusion RBAC re-check in `HybridSearcher` was
   the only thing keeping unauthorised hits out — and it landed twice
   as a security regression (most recently when a cross-org user got
   3 ACME docs back through the BM25-only branch).
2. **Multi-tenancy.** With `SAR_MULTI_TENANT_COLLECTIONS=true` each org
   has its own Qdrant collection, but the BM25 pickle stayed shared
   across all orgs. Two orgs sharing one BM25 index defeats the
   isolation story.

**Decision:**
1. Drop `rank_bm25` and `utils/file_lock.py`. Replace with Qdrant 1.10+
   native sparse vectors stored alongside dense vectors on each point.
2. New `retrieval/sparse_embeddings.py` with two pluggable backends:
   - `bm25` — whitespace tokenize + `zlib.crc32` hash for deterministic
     integer indices, term-frequency normalised by max-tf. Zero new
     dependencies. Default.
   - `splade` — `naver/splade-cocondenser-ensembledistil` via
     `transformers.AutoModelForMaskedLM` with `log(1 + ReLU(x))` +
     max-pool. Requires the `[embeddings-local]` extra. Falls back to
     `bm25` on import/runtime errors.
3. `HybridSearcher` now calls `search_sparse_with_rbac` on
   `QdrantManager`, which applies the same `build_rbac_filter` that
   dense uses. The post-fusion RBAC re-check goes away — sparse-only
   results are already authorised.
4. `scripts/migrate_to_splade.py` walks every point in a collection
   and re-upserts it with both dense and sparse vectors. Idempotent.

**Consequences:**
- (+) The entire cross-tenant / over-clearance / role-mismatch BM25
  bypass class is **structurally impossible**. Sparse search returns
  zero unauthorised candidates by construction.
- (+) Per-tenant isolation: with multi-tenant collections, each org's
  sparse index lives in its own collection alongside its dense index.
- (+) ~200 LOC removed from `HybridSearcher` (the file-lock dance, the
  conditional re-check, the BM25-only fetch branch).
- (+) Benchmark (`evaluation/benchmarks/splade_vs_bm25.md`) shows sparse
  is ~55× faster than dense per query (no Ollama round-trip) on the
  NIST corpus. Hybrid (RRF over both) recovers ~80% of dense quality
  while being resilient to embedding-service outages.
- (-) Sparse latency advantage doesn't compose with dense — hybrid is
  bounded by the slowest path.
- (-) SPLADE requires `[embeddings-local]` (transformers + torch ~2 GB
  install). The default `bm25` backend keeps the slim profile.
- (-) Existing collections need re-indexing via the migration script
  before sparse queries return results.

## ADR-021: LlamaGuard 3 as drop-in escalation backend

**Status:** Accepted (2026-05-21)

**Context:**
ADR-equivalent `guardrails_strict` mode escalated regex-passed queries
to the synth-grade `qwen3:8b` with a free-form "respond SAFE or UNSAFE"
prompt (`core/agents/guardrails_llm.py`). That works but is loose: the
synth model is not fine-tuned for content classification, and any prompt
the model rephrases on its way to a token ends up scored SAFE. Modern
deployments expect a purpose-built classifier with a well-known
taxonomy.

**Decision:**
1. Add `core/agents/guardrails_llamaguard.py` (~165 LOC, including the
   Meta-published chat template + S1-S14 category map). Calls Ollama with
   the `llama-guard3:8b` model (`ollama pull llama-guard3:8b`).
2. New flag `SAR_GUARDRAILS_BACKEND` with values `"llm"` (default,
   legacy) and `"llamaguard"`. Selector lives in
   `core/agents/guardrails.py::guardrails_check`. Regex always runs
   first; only regex-passed queries reach the escalation in strict mode.
3. Parsed category codes (`S1-S14`) map to a stable
   `guardrails_reason` value (e.g. `S2 → non_violent_crimes`). Unknown
   codes degrade to `llamaguard_s<n>` so unparsed model output still
   yields a meaningful audit row.
4. Fail-open on transport errors (Ollama outage, model not pulled). The
   regex gate already ran ahead of us; we never drop user content on
   infrastructure flakes.

**Consequences:**
- (+) Tighter classification — LlamaGuard 3 was fine-tuned for exactly
  this task, and the S1-S14 taxonomy gives recruiters a credible
  "industry-standard guardrail" answer.
- (+) Same Ollama inference path as the rest of the stack. No new
  Python dependency, no new download infra. ~165 LOC of new code total.
- (+) Backward-compatible. `SAR_GUARDRAILS_BACKEND` defaults to `"llm"`
  so existing deployments don't change behaviour without opt-in.
- (-) Adds another model to pull (~5 GB). For deployments that don't
  enable strict mode, this cost is never paid.
- (-) Category granularity exposes more vocabulary in the audit log
  (the previous binary `llm_escalation_unsafe` was simpler to grep).
  Net win because the categories let policy reviews trace blocked
  queries to the specific Meta policy clause that fired.

## ADR-022: Fine-tuned domain reranker as opt-in checkpoint

**Status:** Accepted (2026-05-23) — fine-tune trained, **+1.60pp NDCG@10 lift on MS-MARCO 500-pair hold-out** (0.7744 → 0.7904), exceeds the ≥1pp bar. Bench report at `evaluation/benchmarks/reranker_finetune.md`.

**Context:**
The retrieval stack ships three reranker modes today (none / cross-encoder /
ColBERTv2). All three use off-the-shelf checkpoints — strong baselines but
not aware of the project's actual document corpus (NIST AI RMF policy
text + ACME synthetic docs). On in-domain queries the off-the-shelf
rerankers can rank a tangentially-relevant chunk above the one a SME
would pick.

**Decision:**
1. Add a `fine_tuned` mode to `settings.reranker_type` plus a new flag
   `SAR_FINETUNED_RERANKER_PATH` (default `data/checkpoints/reranker-domain-v1`).
   The factory in `core/agents/retriever.py::_get_reranker` loads the
   path via the existing `retrieval.reranker.Reranker` class —
   sentence-transformers' `CrossEncoder` already loads from a local
   directory the same way it loads from HuggingFace.
2. New `scripts/train_reranker.py` (~220 LOC). Pulls MS-MARCO Passage
   Ranking small split from HuggingFace, optionally mines hard negatives
   from the local Qdrant index, fine-tunes from BGE-Reranker-v2-M3 with
   `CrossEncoder.fit`. Writes the checkpoint and a `train_meta.json`
   alongside it for reproducibility.
3. New `scripts/bench_reranker.py` (~200 LOC). Computes NDCG@10 hold-out
   on MS-MARCO and on the optional `evaluation/nist_rerank_gold.jsonl`
   in-domain gold set. Writes a Markdown report at
   `evaluation/benchmarks/reranker_finetune.md` + a timestamped JSON.
   Acceptance bar: fine-tuned must beat baseline by ≥1pp on MS-MARCO
   and win on the NIST gold set.

**Training run details (2026-05-23):**
Trained on RTX 3060 12GB: 100k MS-MARCO Passage Ranking rows (200k
InputExamples after pos/neg pairing), 1 epoch, batch 16, AMP fp16 via
`use_amp=torch.cuda.is_available()`. Wall time 14,090s (~3:55 hr) at
14.2 samples/sec. Final training loss 0.4983. No hard-negative mining
(the 476-doc local NIST Qdrant corpus mismatches MS-MARCO queries so
mined negatives would be lower-signal than MS-MARCO's own random
negatives). Checkpoint `data/checkpoints/reranker-domain-v1/`
(2.27 GB safetensors, gitignored via `data/*`). Bench artefacts:
`evaluation/benchmarks/reranker_finetune.md` + timestamped JSON.

NIST in-domain bench arm is still pending — needs the hand-labelled
`evaluation/nist_rerank_gold.jsonl` (separate P3 roadmap item). The
bench script gracefully skips the NIST arm when the file is absent.

**Consequences:**
- (+) Domain-specific reranker is now a one-flag flip away. The
  retrieval factory + bench harness + training script are all in tree.
- (+) Reproducible: `train_meta.json` next to every checkpoint records
  base model, sample sizes, epochs, hard-negative mining flag.
- (+) Bench works against any cross-encoder, so even without running our
  own training we can compare BGE-Reranker-v2-M3 vs CrossEncoder-MiniLM
  vs anyone else with one command.
- (-) Actual fine-tuned checkpoint is not committed — 2.27 GB
  safetensors changes per training run, gitignored via `data/*`.
  Users must run `scripts/train_reranker.py` themselves before
  flipping `SAR_RERANKER_TYPE=fine_tuned`. Reproducibility metadata
  lands in `data/checkpoints/<name>/train_meta.json`.
- (-) `[embeddings-local]` extra is required (sentence-transformers +
  torch ~2GB). Same cost as the existing cross_encoder mode, so this is
  not a new tax for users who already opted into local rerankers.

## ADR-023: Threshold calibration against a labelled gold set

**Status:** Accepted (2026-05-23)

**Context:**
`settings.confidence_threshold` and `settings.faithfulness_threshold` were
picked by intuition at 0.6 / 0.7. They gate two important pipeline
behaviours — `needs_human_review` (UI surface) and the NLI faithfulness
decision (annotate-or-drop unsupported sentences). Intuition-picked
cut-offs mean a real shift in the upstream stack (new reranker, new
guardrails backend, model upgrade) silently changes what counts as
"low-confidence" without anyone noticing. Pre-req for shipping
confidence-driven UX (rejecting answers, escalating to human review,
auto-routing low-confidence queries to cloud).

**Decision:**

1. Hand-label a 50-row gold set at `evaluation/golden_set.jsonl`. Each
   row carries `expected_confidence_band` and `expected_faithfulness_band`
   ("high"/"medium"/"low") plus `expected_outcome` ("answer"/"refuse"/
   "block"). Coverage:
   - NIST AI RMF factual + inferential (12)
   - ACME synthetic RBAC corpus — public, engineering, finance (15)
   - RBAC negative tests, including cross-org External (6)
   - Out-of-scope questions (5)
   - Prompt-injection probes (5)
   - Bilingual Arabic queries (2)
   - Adversarial / unsupported-claim probes (5)
2. New `scripts/calibrate_thresholds.py` runs every gold row through the
   live RAG pipeline (with `SAR_FAITHFULNESS_GATE_ENABLED=true` and a
   bumped `SAR_REQUEST_TIMEOUT_S` so the NLI gate has room to finish),
   records the `(confidence_score, faithfulness_ratio)` pair per row,
   sweeps thresholds across `[0.0, 1.0]` in 0.05 steps, and picks the
   value maximising Youden's J (`TPR - FPR`) — the cut-off that best
   separates positive-band rows from negative-band rows. Blocked rows are
   excluded from the faithfulness sweep so their default ratio of 1.0
   doesn't pollute the negative tail.
3. Chosen thresholds + full sweep curves persist to
   `evaluation/calibration.json`. `config/settings.py::_apply_calibration`
   reads that file at import and updates the runtime thresholds — but
   only when the env var is unset, so operators can still pin per
   deployment via `SAR_CONFIDENCE_THRESHOLD` / `SAR_FAITHFULNESS_THRESHOLD`.
4. The script also writes a measured baseline (Ragas if installed,
   lexical-overlap fallback otherwise) into `evaluation/baseline.json`,
   replacing the legacy hand-picked numbers. `evaluation/nightly.py`
   already compares against that file and fails the build on a >5pp
   drop, so calibration runs are simultaneously the new "good
   known-state" for nightly regression detection.

**Consequences:**

- (+) Thresholds are now data-driven against real RBAC + retrieval
  paths, not a guess. Calibration JSON keeps the full curve so a future
  reviewer can pick a different operating point (e.g. higher precision)
  without re-running the pipeline.
- (+) Single source of truth: settings.py reads calibration.json once
  at import — no code-path branching elsewhere. Env override still wins,
  preserving the "operator can pin anything" principle.
- (+) Nightly CI gates on the same measured baseline that calibration
  emits — `>5pp` drop fails the build via the existing
  `evaluation.nightly` workflow.
- (+) Reproducible: each run writes a timestamped snapshot under
  `evaluation/results/calibration_<ts>.json` containing per-row
  pipeline outputs, so any future re-pick can be done offline via
  `scripts.calibrate_thresholds --from-results <path>` without
  re-running ~80 minutes of live pipeline.
- (-) Gold set is 50 rows. Larger (~200-500) would tighten confidence
  intervals on the chosen threshold; current bar is "is the cut-off
  obviously in the right region" not "what is the exact optimum."
- (-) Calibration takes ~80-120 minutes on local Ollama because the
  faithfulness gate adds per-sentence LLM calls. Not in CI by default;
  re-run when the upstream model or reranker changes.
