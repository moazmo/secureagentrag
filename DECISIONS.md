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

**Status:** Accepted (2026-05-23) — fine-tune trained, **+1.60pp NDCG@10 on MS-MARCO 500-pair hold-out** (0.7744 → 0.7904) **and +0.54pp on the 20-row NIST in-domain gold** (0.9162 → 0.9215). Both ADR acceptance criteria (≥1pp on MS-MARCO + win on NIST) met. Bench report at `evaluation/benchmarks/reranker_finetune.md`; NIST gold at `evaluation/nist_rerank_gold.jsonl`.

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

## ADR-024: Per-tenant SPLADE isolation + manager cache

**Status:** Accepted (2026-05-23)

**Context:**
ADR-020 moved the BM25 / SPLADE sparse field inside Qdrant alongside the
dense vector — eliminating the cross-tenant bypass class that the legacy
``rank_bm25`` pickle exposed. Multi-tenant mode
(``SAR_MULTI_TENANT_COLLECTIONS=true``) routes each org to its own
collection ``documents_{org_id}``, and ``QdrantManager.ensure_collection``
provisions the named sparse field on every collection. The sparse path
was therefore already structurally isolated per tenant.

Two issues remained:

1. **Per-request manager rebuild.** ``QdrantManager.for_org(org_id)`` constructed
   a fresh ``QdrantManager`` (= new ``QdrantClient`` HTTP pool + extra
   ``get_collections`` round-trip via ``ensure_collection``) on every
   request when running in multi-tenant mode. At >1 req/sec that
   produced a measurable Qdrant overhead and grew the HTTP client pool
   unbounded.
2. **Regression coverage was thin.** The existing
   ``test_qdrant_manager_for_org_returns_new_in_multi_tenant`` only
   verified that ``for_org`` did not return ``self``. Nothing pinned
   "each tenant gets its own sparse field" or "repeat calls reuse the
   same manager".

**Decision:**

1. Cache per-tenant ``QdrantManager`` instances on the root manager via
   a ``self._tenant_cache: dict[str, QdrantManager]`` keyed by the
   resolved org-specific collection name. First ``for_org(org_id)`` call
   constructs + ``ensure_collection`` once; subsequent calls are O(1)
   dict lookups. The cache stays bound to *this* root manager, so
   distinct roots (e.g. different ``SAR_QDRANT_URL`` overrides) keep
   distinct caches.
2. Update the docstring on ``for_org`` to state the structural-isolation
   guarantee explicitly: "Qdrant cannot scan across collections in a
   single query, so a query bound to org B's manager never sees org A's
   sparse data even if the field name is identical".
3. Add three new tests in ``tests/test_retrieval/test_multitenancy.py``
   pinning the contract:
   - ``test_qdrant_manager_for_org_caches_per_tenant_managers`` —
     repeat calls return the same instance; distinct orgs produce
     distinct managers; ``__init__`` + ``ensure_collection`` each run
     exactly once per distinct tenant.
   - ``test_qdrant_manager_ensure_collection_creates_sparse_field`` —
     ``create_collection`` is called with a ``SparseVectorParams`` entry
     under ``settings.sparse_vector_name`` (regression guard against
     someone deleting the sparse slot in a future refactor).
   - ``test_qdrant_manager_for_org_distinct_tenants_isolated`` — two
     orgs resolve to distinct collection names.

**Consequences:**

- (+) Multi-tenant request latency drops by one ``get_collections``
  round-trip per request (typically 5-15 ms on local Qdrant, larger on
  cloud). At sustained load this is the difference between a memory-
  bounded client pool and an HTTP-connection leak.
- (+) Sparse isolation is now pinned by regression tests, not just by
  the structural argument in ADR-020. Future refactors that accidentally
  drop the sparse field or share a manager across tenants will fail
  the build.
- (+) The cache lives on the root manager — a new
  ``QdrantManager(...)`` instance gets a fresh cache, so test fixtures
  that build their own root remain isolated.
- (-) The cache has no TTL or LRU eviction. For long-running processes
  with thousands of tenants the per-process memory grows linearly with
  the number of orgs seen. Not a concern at current scale; flag for
  revisit if active-tenant cardinality crosses ~1000.
- (-) Cache invalidation on schema change requires a process restart.
  Acceptable because schema changes are rare and the existing
  ``scripts/migrate_to_splade.py`` already requires a full re-ingest.

---

## ADR-025: BYOK Demo Mode (per-request key + session collections)

**Date:** 2026-05-26
**Status:** Accepted (flipped from draft when `/byok/chat` served traffic
end-to-end from Egypt via the HF Space)

**Context:**
The public production launch needs a demo any visitor can try without us
paying per-token LLM costs. Three patterns were considered:

1. Owner-funded — single Groq key paid by the owner. Free tier exhausts
   in hours under any traffic. Rejected.
2. Per-user OAuth + cloud credits — sign visitors up for N free queries.
   30 s of OAuth before the demo is visible. Friction kills the demo.
   Rejected.
3. **BYOK — visitor pastes their own key** — zero key-burn risk on the
   owner side, introduces the threat surface of routing visitor keys
   through our backend. Must never persist.

Independent constraint: HF Spaces is the chosen backend host (ADR-026).
Its ephemeral disk + 48-hour idle sleep forces a stateless-per-session
design that matches BYOK naturally.

**Decision:**
Implement a BYOK mode behind ``SAR_BYOK_MODE=true``. Concretely:

- Per-request headers ``X-User-LLM-Key``, ``X-User-Provider``,
  ``X-User-Ollama-URL``, ``X-Session-ID``, ``X-Demo-Persona`` are extracted
  by ``interfaces/byok.py`` and threaded into the LangGraph pipeline.
- Owner-key fallback is gated by a per-IP rate limit. Default
  ``SAR_BYOK_OWNER_KEY_QUOTA_PER_HOUR=10`` (raised from initial 3 after
  live testing showed visitors blocked on the 3rd query).
- ``X-Forwarded-For`` leftmost token is the trusted IP because HF Spaces
  masks ``request.client.host`` behind a reverse proxy. Without this,
  every visitor would share one throttle bucket.
- Each session gets a Qdrant collection named
  ``documents_sess_<sanitized_session_id>`` for uploads (see ADR-029).
- A purge cron deletes session collections older than
  ``SAR_SESSION_COLLECTION_TTL_HOURS=24``.
- Phoenix instrumentation is forcibly disabled when BYOK mode is on so
  no observability layer sees the visitor key.
- Audit log redacts API key patterns (Groq ``gsk_*``, OpenAI ``sk-*``
  and ``sk-proj-*``, Anthropic ``sk-ant-*``, HuggingFace ``hf_*``,
  Vercel ``vcp_*``, Qdrant JWT) before persist.
- CORS allowlist limited to the Vercel frontend URL via
  ``SAR_CORS_ALLOW_ORIGINS``.
- Persona presets ``_DEMO_PERSONAS`` (engineer / compliance / executive)
  map ``X-Demo-Persona`` to clearance + roles + system-prompt style.
- BYOK mode bypasses the LLM-evaluator and grader by default
  (``SAR_BYOK_SKIP_GRADER=true``, ``SAR_BYOK_SKIP_EVALUATOR=true``) — see
  ADR-030 for the cost-trade rationale.
- BYOK mode suppresses the synth-side sensitivity disclaimer because the
  frontend renders a dedicated ``sensitivity:`` badge instead.

**Consequences:**

- (+) Zero recurring LLM cost regardless of demo traffic.
- (+) Recruiter can paste a ``$0.10`` test-budget Groq key and verify
  the entire stack works without giving us anything.
- (+) Per-session collection isolation is a free side-effect that
  doubles as a multi-tenancy proof point.
- (+) Aligns with the privacy-first narrative — "we never store your
  keys, even for telemetry."
- (-) Introduces a new threat surface (BYOK key in transit). Mitigated
  by HTTPS-only, never-log-body, redaction regression test, and the
  no-Phoenix rule.
- (-) Owner-key fallback still exists for visitors without a key; 10/hr
  per IP throttle defends the daily Groq budget. Rotation declined —
  the throttle structurally caps single-abuser cost below the budget.
- (-) BYOK mode changes the *meaning* of half the existing settings
  (faithfulness, evaluator, RAG-fusion, reranker can all be off in
  production while the codepaths remain in tree). ADR-030 documents
  those trade-offs explicitly.
- (-) Session cleanup adds a cron job that must stay alive.

**Acceptance criteria (all met 2026-05-26):**

- ✅ ``interfaces/byok.py`` shipped, owner + visitor BYOK paths green
- ✅ ``tests/test_security/`` API-key redaction tests green
- ✅ ``tests/test_interfaces/test_byok.py`` per-IP throttle + header
  extraction tests green
- ✅ Session collection purge tested live on Qdrant Cloud
- ✅ Live ``/byok/chat`` round-trip from Egypt streaming SSE tokens

---

## ADR-026: Hugging Face Spaces as Production Backend Host

**Date:** 2026-05-26
**Status:** Accepted (flipped from draft when ``/healthz`` returned 200
from Egypt and ``/byok/chat`` completed end-to-end)

**Context:**
Production backend host must hit four hard constraints: zero USD, no
credit card at signup, available in Egypt, no Render-style cold start.

A 2026-05-25 free-tier audit narrowed the candidates to Hugging Face
Spaces (Docker, CPU Basic, 16 GB RAM, 48 h idle sleep), Northflank
Sandbox (CC requirement unclear), Streamlit Community Cloud (shorter
sleep), and Vercel Hobby Python (60 s function timeout — too short).

**Decision:**
Use **Hugging Face Spaces with the Docker SDK on CPU Basic hardware**.

- Space: ``huggingface.co/spaces/LeomordKaly/secureagentrag-api``
- Public URL: ``LeomordKaly-secureagentrag-api.hf.space``
- Hardware: CPU Basic (2 vCPU, 16 GB RAM, $0/mo)
- Build: ``Dockerfile.hf`` in the GitHub repo, two-stage build with uv
- Sleep mitigation: GitHub Actions cron at 03:17 UTC daily pings
  ``/healthz`` plus a tiny ``/byok/chat`` round-trip
- Secrets: ``SAR_QDRANT_URL``, ``SAR_QDRANT_API_KEY``, ``SAR_GROQ_API_KEY``
  via HF Space secrets panel — never baked into the image
- Reranker: ``SAR_RERANKER_TYPE=none`` in production. The fine-tuned 2.3
  GB checkpoint is intentionally not uploaded (would blow CPU Basic disk;
  ADR-022 acceptance criteria are still met in the available mode)

**Consequences:**

- (+) Zero recurring cost.
- (+) 16 GB RAM fits BGE-M3 + FastAPI + Python stack with >10 GB headroom.
- (+) ``git push`` deploy via ``scripts/deploy_hf_space.py``.
- (+) Free ``.hf.space`` subdomain — no domain purchase required.
- (+) 48-hour sleep boundary (vs Render's 15 min) — 192× more forgiving.
- (-) 30–60 s cold start on first wake. Mitigated by keepalive cron.
- (-) Ephemeral disk — audit log + checkpoints wiped per restart.
  Acceptable for BYOK demo; aligns with the privacy story.
- (-) CPU only — synth latency on Groq dwarfs anything CPU does locally,
  so this is a non-issue in practice.
- (-) Custom domain requires HF Pro ($9/mo). Skipped — the Vercel
  subdomain is the recruiter-facing URL.

**Acceptance criteria (all met 2026-05-26):**

- ✅ HF Space provisioned, reachable from Egypt at 0.54 s TTFB
- ✅ ``Dockerfile.hf`` builds cleanly (450 s on the HF runner)
- ✅ ``curl .../healthz`` returns 200 from Egypt
- ✅ Phase 2 BYOK backend runs successfully on the Space
- ✅ Live ``/byok/chat`` round-trip with owner-key + with visitor BYOK key

---

## ADR-027: Vercel + Next.js 16 Frontend (drop Streamlit for the public demo)

**Date:** 2026-05-26
**Status:** Accepted (flipped from draft when the Vercel deploy served a
streaming response from Egypt end-to-end)

**Context:**
The Streamlit UI (``app/``) is the local-dev face but recruiter optics +
mobile responsiveness + 2026 visual standards push us to Next.js +
shadcn/ui for the public demo. Streamlit-on-HF-Spaces also has websocket
reliability issues through HF's proxy.

**Decision:**
Build a **Next.js 16 App Router frontend with Tailwind v4 + SSE
streaming**, deployed to **Vercel Hobby plan**. Streamlit remains in the
repo for local development; it is no longer the public face of the demo.

- Sibling repo: ``github.com/moazmo/secureagentrag-web``
- URL: ``secureagentrag-web.vercel.app`` (Hostinger custom-domain detour
  cancelled 2026-05-27)
- Streaming: Vercel Edge runtime proxy with ``duplex: "half"`` so the
  upstream SSE body pipes through without buffering. Token deltas reach
  the browser sub-100 ms.
- BYOK input: drawer + localStorage persistence (never cookies — CSRF
  surface)
- Audit viewer: client-side download as ``.jsonl`` with the SHA-256
  ``prev_hash``/``entry_hash`` chain intact for offline verification
- Persona switcher: three preset RBAC profiles (engineer / compliance /
  executive) — header drives the backend persona map
- PDF upload: drag-drop + progress bar → multipart POST to backend, 5 MB
  / 5 files / 60 chunks per file cap (ADR-029)
- Theme: eye-comfort dark palette (``--background: #0c0d10``,
  ``--foreground: #e6e7ea``, ``--accent: #6b8afd``)

**Consequences:**

- (+) Zero cold start (Vercel Edge for SSR + Edge function for the SSE
  bridge).
- (+) "Next.js on Vercel" matches the recruiter mental model for AI
  products. Streamlit signals "data-scientist tool."
- (+) BYOK UX is clean — drawer + localStorage is a familiar pattern.
- (+) Mobile responsiveness comes free from Tailwind primitives.
- (+) Separates frontend and backend lifecycles — UI can iterate without
  redeploying the model layer.
- (-) Two repos to maintain.
- (-) Some duplication of types between TypeScript and Python; mitigated
  by mirroring the Pydantic schema by hand for the demo surface (we are
  small enough that an openapi codegen pipeline would be overkill).
- (-) The SSE wire format had to be bridged because Vercel AI SDK's
  default contract differs from our ``graph.astream(stream_mode=
  ["updates","custom"])`` events. ~120 LOC of bridge code in
  ``secureagentrag-web/src/lib/stream.ts``.

**Acceptance criteria (all met 2026-05-26..27):**

- ✅ Sibling repo created on GitHub
- ✅ Vercel project linked + ``vercel --prod`` deploy clean
- ✅ ``https://secureagentrag-web.vercel.app`` reachable from Egypt
- ✅ BYOK drawer saves to localStorage and forwards header on next request
- ✅ End-to-end streaming smoke against live HF Space backend works
- ✅ Lighthouse acceptable on mobile

---

## ADR-028: Qdrant Cloud Free Tier + Per-Session Collections

**Date:** 2026-05-26
**Status:** Accepted (flipped from draft when the demo corpus ingest +
RBAC matrix proved live on the Cloud cluster from the HF Space)

**Context:**
For the HF Spaces backend (ADR-026), self-hosting Qdrant inside the Space
is risky because the disk is ephemeral. We need an externally-hosted
Qdrant that stays alive across HF Space restarts and supports both the
RBAC payload filter and the native sparse field shape from ADR-020/ADR-024.

**Decision:**
Use **Qdrant Cloud free tier (1 GB / 1M vectors / always-on)**.

- Cluster: 1 free node in AWS us-east-1 (~150 ms latency from Egypt)
- Base collection: ``documents`` — 10 demo RBAC docs (138 chunks, 276
  points incl. sparse), tagged engineer / compliance / executive
- Per-session: ``documents_sess_<sanitized_session_id>`` (see ADR-029)
- TTL: 24 hours via ``SAR_SESSION_COLLECTION_TTL_HOURS``
- Purge: ``retrieval/session_purge.py::purge_expired_sessions`` deletes
  session collections past TTL; ``schedule_session_purge`` runs it every
  6 h via APScheduler in the FastAPI lifespan
- Credentials: ``SAR_QDRANT_URL`` + ``SAR_QDRANT_API_KEY`` as HF Space
  secrets
- Sparse: BGE-M3 dense + BM25 sparse (no SPLADE in production — keeps
  cold path zero-dep)
- ``SAR_MULTI_TENANT_COLLECTIONS=true`` so the base collection route
  flows through the same ``for_org()`` path as the per-session ones

**Consequences:**

- (+) Always-on, no sleep on the vector store.
- (+) Zero migration — the existing ``retrieval/qdrant_client.py``
  already speaks to remote URLs.
- (+) Per-session isolation is structural: each query targets exactly
  one ``documents_sess_<sid>`` plus the base, both under the same RBAC
  filter.
- (+) The 24 h TTL is a clean answer to "what happens to my uploads".
- (-) 1 GB cap. At peak (~50 concurrent visitors × 5 files × 60 chunks)
  ~120 MB resident — comfortable. Purge cron is the safety net.
- (-) Network round-trip from HF Space → Qdrant Cloud adds ~30–50 ms.
  Dwarfed by LLM latency.
- (-) Cluster URL embedded in HF Space secrets — rotation requires
  updating the Space config.

**Acceptance criteria (all met 2026-05-26):**

- ✅ Qdrant Cloud free cluster provisioned and accessible from HF Space
- ✅ Phase 1c sparse vector smoke green
- ✅ Phase 7 demo corpus ingest (10 docs / 138 chunks) live
- ✅ Purge cron tested against artificial old collections

---

## ADR-029: BYOK Document Uploads — Session-Scoped Qdrant Collections + Dual-Collection RRF Retrieval

**Date:** 2026-05-27
**Status:** Accepted

**Context:**
Visitors want to bring their own documents into the demo (resume,
runbook, paper, etc.) and ask questions across them. Three patterns:

1. Re-ingest into the base ``documents`` collection — would pollute the
   shared corpus and require RBAC trickery to keep visitor docs scoped.
   Rejected.
2. Per-org collection via existing ``for_org()`` — assumes an authenticated
   org_id which BYOK visitors don't have. Rejected.
3. **Per-session collection** keyed on ``X-Session-ID``, with retrieval
   fanning out to both the base collection and the session collection,
   results fused by RRF. **Chosen.**

**Decision:**

- ``QdrantManager.for_session(session_id)`` mirrors ``for_org()``: caches
  per-session managers, creates ``documents_sess_<sid>`` with the same
  payload index set (``org_id``, ``sensitivity_level_int``, ``roles``,
  ``user_id``, ``source_file``, ``source_file_id``) so the RBAC filter
  shape is identical.
- New endpoints in ``interfaces/api.py``:
  - ``POST /byok/uploads`` — multipart upload, runs ``ingestion/pipeline``
    against the session collection, returns ``{file_id, chunks, size_bytes}``
  - ``GET /byok/uploads`` — list visitor's files
  - ``DELETE /byok/uploads/{file_id}`` — delete by ``source_file_id``
- Hard caps (env-tunable):
  - ``SAR_BYOK_UPLOAD_MAX_BYTES=5*1024*1024`` (5 MB per file)
  - ``SAR_BYOK_UPLOAD_MAX_FILES=5`` (per session)
  - ``SAR_BYOK_UPLOAD_MAX_CHUNKS_PER_FILE=60`` (chatty PDFs rejected)
  - ``SAR_BYOK_UPLOAD_ALLOWED_EXTENSIONS=[".txt", ".md", ".pdf"]``
- ``HybridSearcher.search(session_id=...)`` runs parallel dense + sparse
  against both base and session collections, RRF-fuses the four result
  sets, returns top_k. The session collection ``top_k`` is bounded by
  ``SAR_TOP_K`` (not ``*2``) to keep candidate count tight.
- Pre-existing bug fixed during this work: ``HybridSearcher._embeddings``
  attribute was actually ``_embedder`` (referenced incorrectly in two
  call sites in ``interfaces/api.py``).

**Consequences:**

- (+) Structurally impossible cross-session leakage — each session's
  collection name carries the session id; one query cannot scan two
  sessions.
- (+) Same RBAC filter applies to both collections (defense in depth
  preserved).
- (+) The 5 MB / 5 file / 60 chunk caps protect the 1 GB Qdrant Cloud
  ceiling and the free-tier CPU budget.
- (+) Frontend gets file-level granularity (list / delete by ``file_id``).
- (-) Two extra round-trips per chat (dense + sparse against the session
  collection). On a 0–5 file session this is negligible — ``top_k=10``
  bound keeps it tight.
- (-) Vercel Edge 30 s timeout is shorter than backend
  ``SAR_REQUEST_TIMEOUT_S=180``. On long pipelines the Edge cuts first
  and returns HTML; ``secureagentrag-web/src/lib/uploads.ts`` does a
  text-then-parse fallback so the user sees an actionable error rather
  than a JSON parse exception.
- (-) Per-file chunk cap (60) rejects long PDFs. Trade-off chosen over
  silent truncation — visitor gets a 422 with the chunk count so they
  know to split the doc.

**Acceptance criteria (all met):**

- ✅ Upload + chat end-to-end smoke on the live demo (Discrete Mathematics
  PDF 56 chunks + NIST AI RMF PDF 135 chunks both ingested and queried)
- ✅ 6 MB file rejected with clear error
- ✅ Delete-by-file_id removes points from the session collection
- ✅ Cross-session isolation pinned by tests
- ✅ Session purge cron deletes 24 h+ collections

---

## ADR-030: Free-Tier Groq Cost Optimisations (BYOK-mode pipeline cuts)

**Date:** 2026-05-27
**Status:** Accepted

**Context:**
Initial BYOK launch fired **5–6 Groq calls per chat** against the free
tier's **30 RPM / 14,400 RPD / 6,000 TPM** budget. A single chatty answer
exhausted the RPM bucket, the next visitor 429-ed, and the answer text
on screen was "[Error generating response]". The Groq console showed the
spike: one chat → ~30 calls/min as multiple visitors stacked.

Per-call breakdown before the fix:

1. Router classifier (1 call)
2. RAG-fusion query reformulation (1 call → 3–5 parallel Qdrant searches)
3. Grader LLM (1 call per candidate doc)
4. Synthesizer (1 call, streaming)
5. Faithfulness gate (1 call per cited sentence — 5–10 extra calls)
6. Evaluator LLM (1 call)

**Decision:**
Pin the Groq model + disable the LLM nodes that don't materially help on
a 10-doc demo corpus. Five env-var-controlled changes baked into
``Dockerfile.hf``:

- ``SAR_GROQ_MODEL=llama-3.1-8b-instant`` — was hardcoded
  ``llama-3.3-70b-versatile``. The 70b model has lower TPM headroom and
  slower throughput (heavier generation = more wall-clock per request);
  8b finishes ~1 s on prompts under 4k tokens with comparable answer
  quality on this corpus. ``inference/router.py::_get_model_for_provider``
  now reads from ``settings.groq_model``.
- ``SAR_RAG_FUSION_ENABLED=false`` — RAG-fusion fires 1 extra Groq call
  per chat to generate N reformulations + N parallel Qdrant searches.
  Useless on a 10-doc corpus where the original query already retrieves
  the right chunks.
- ``SAR_BYOK_SKIP_EVALUATOR=true`` (default in BYOK mode) — the evaluator
  LLM call is replaced by a heuristic ``confidence = citation_coverage *
  0.5 + evidence_strength * 0.5``. Saves 1 Groq call. The LLM evaluator
  remains on the codepath; flag flip restores it.
- ``SAR_BYOK_SKIP_GRADER=true`` (default in BYOK mode) — the grader's
  per-document LLM relevance score is skipped; retrieved docs auto-marked
  relevant. Saves N–10 calls. Loose relevance threshold
  (``SAR_RELEVANCE_THRESHOLD=0.55`` / ``_RETRY=0.3``) keeps the corrective
  loop active for genuine misses without LLM judgment.
- ``SAR_FAITHFULNESS_GATE_ENABLED=false`` — gate makes 5–10 extra calls
  per answer. The synthesizer's own citation discipline (mandatory inline
  ``[N]`` markers + sources-only prompt) is strong enough for the demo.
- Router classifier shortcut: queries ≤80 chars in BYOK mode skip the
  classifier and short-circuit to ``query_type="simple"``. Saves 1 call
  on the common "what is X?" / "how do I Y?" case.
- ``SAR_MAX_RETRIES=1`` — cap the corrective-RAG retry loop. Two refines
  is enough on a 10-doc corpus; further rewrites stack Groq calls without
  meaningfully improving recall.
- ``SAR_RERANK_TOP_K=10`` (was 5) — with the LLM grader bypassed, this
  doubles as the synth doc budget. Bigger context here is the easiest
  quality lever now that the reranker is off.
- ``SAR_RERANKER_TYPE=none`` — the fine-tuned reranker is intentionally
  off in production (no checkpoint on the HF Space; on a 10-doc corpus
  the cross-encoder cold-load tax + top-5 cut routinely drops the
  visitor's own chunk).

**After the cuts:**

- ~2 Groq calls per chat (router shortcut + synth). Worst case ~3 with
  router classifier or 1 rewrite retry.
- Live verification: kubectl rollback query returns full step-by-step
  runbook with ``[9]``/``[10]`` citations, 13 s latency, ``groq · 8b``
  badge, 2 citations panel, no sensitivity disclaimer, no 429.

**Consequences:**

- (+) Demo survives sustained traffic on the 30 RPM Groq budget.
- (+) ``llama-3.1-8b-instant`` is fast enough that TTFB feels snappy
  on Egypt's 200 ms RTT.
- (+) Each toggle is a single env var; ``flag-flip`` to a paid Groq tier
  or local Ollama deploy restores the full pipeline without a code
  change.
- (-) Faithfulness gate is off in production. ADR claim "NLI gate
  enforces citation entailment" is true *in self-hosted mode*; the live
  demo trades this for cost. The frontend renders citation chips but
  doesn't show ``*[unsupported]*`` annotations.
- (-) LLM grader is bypassed. Retrieved docs that are weakly relevant
  reach synth; the synthesizer's "answer only from sources or refuse"
  prompt is the only guard against irrelevant context. Holds on the demo
  corpus; would not hold at scale.
- (-) RAG-fusion off. Visitors with under-specified queries get fewer
  reformulation chances. Acceptable on a small corpus.
- (-) Reranker off. Top-K relies on dense + BM25 RRF only. Acceptable
  for ≤200 doc corpora per ADR-022 bench data.

**Acceptance criteria (all met):**

- ✅ Same chat that 429-ed before now succeeds end-to-end in 13 s
- ✅ Groq console shows ~2 calls / chat on the production trace
- ✅ Test fixtures updated for the new bypass paths (``mock_settings.
  rerank_top_k`` raised to 20, ``mock_settings.groq_model`` /
  ``openai_model`` / ``anthropic_model`` added to router patches)
- ✅ All tests green (623 pass / 3 skip / 626 collected; CI gates with `--extra api`)
- ✅ Live recorded screenshot of a successful Q+A with citations

---

## ADR-031: Prometheus/Grafana Metrics Layer (self-hosted, BYOK-safe)

**Date:** 2026-05-28
**Status:** Accepted

**Context:**
The project already had Arize Phoenix / OpenTelemetry *tracing* — per-LLM-call
spans capturing prompts, completions, and latency. Tracing is the right tool
for "what happened in this one request," but it is *hard-disabled under BYOK*
(``utils/observability.py::setup_tracing`` short-circuits when
``settings.byok_mode``) because spans would capture a visitor's keys-in-context
and uploaded text. That left the public demo with **no aggregate operational
view** — no answer to "what's the p95 pipeline latency," "how often does the
guardrails gate fire," or "what fraction of traffic routes to Groq vs Ollama."

A friend's repo praised by an "AI engineering expert" shipped a
Prometheus + Grafana + exporters stack as a headline signal. We had the
harder substance (RBAC, faithfulness, audit chain) but were missing the
legible operational dashboard that reviewers skim for first.

**Decision:**
Add a second, complementary observability layer: **aggregate metrics** via
Prometheus, visualised in Grafana. Crucially, metrics are *counters and
histograms only* — no request content ever enters a label — so unlike
tracing they are privacy-safe even under BYOK.

- ``utils/metrics.py`` — zero-hard-dependency module. Imports
  ``prometheus_client`` behind a try/except; without the ``[metrics]`` extra
  every recorder is a no-op and ``METRICS_ENABLED`` is ``False``. Four custom
  RAG metrics on the global registry:
  ``rag_pipeline_latency_seconds`` (histogram, ``outcome`` label, buckets to
  the 180 s SLO), ``rag_pipeline_requests_total`` (counter, ``outcome``),
  ``guardrails_blocked_total`` (counter, ``gate`` + ``reason``),
  ``inference_routed_by_provider_total`` (counter, ``provider``), and
  ``faithfulness_dropped_total`` (counter).
- Label cardinality is bounded: providers clamp to a known set or ``other``;
  guardrails reasons clamp to a controlled vocabulary or ``other`` — an
  attacker-controlled rejection string can never explode the series count.
- ``record_pipeline_run(state, latency_ms)`` is called at **all four** pipeline
  terminal points in ``core/graph.py`` (sync success, sync timeout, streaming
  final, streaming timeout). It derives the outcome from the final state
  (``blocked`` / ``timeout`` / ``review`` / ``success``) and never raises.
- ``interfaces/api.py`` mounts ``prometheus-fastapi-instrumentator`` for
  HTTP-level metrics and serves ``/metrics`` from the shared default registry;
  if the extra is absent it falls back to a manual ``/metrics`` route that
  501s. Either way the custom RAG metrics appear in the same exposition.
- ``docker-compose.observability.yml`` overlays Prometheus + Grafana + a
  uvicorn ``api`` service (built with ``INSTALL_EXTRAS=[api,metrics,...]`` via a
  new Dockerfile ARG). Grafana auto-provisions the datasource
  (``deploy/grafana/provisioning/``) and the "SecureAgentRAG — RAG Pipeline"
  dashboard (``deploy/grafana/dashboards/secureagentrag.json``).

**Consequences:**

- (+) Self-hosted deploys get a real operational dashboard: pipeline latency
  percentiles, request outcomes, provider routing mix, guardrails blocks, and
  faithfulness drops — the legible signal reviewers look for.
- (+) Privacy-safe by construction: aggregate-only, so it can run even where
  Phoenix tracing cannot. Two layers with a clear split (tracing = per-request
  detail, self-hosted-only via Phoenix; metrics = aggregate, always-safe).
- (+) Zero cost on the public demo: the HF Space ships without the ``[metrics]``
  extra, so ``/metrics`` is a 501 no-op and no collector runs. No new attack
  surface, no memory overhead on CPU Basic.
- (−) The Grafana stack is self-hosted only — the free HF Space CPU tier can't
  host Grafana, so the public demo has no live dashboard (README documents the
  local ``docker compose`` path instead).
- (−) HTTP metrics depend on ``prometheus-fastapi-instrumentator``; the manual
  fallback exposes only the custom RAG metrics, not per-handler HTTP latency.

**Acceptance criteria (all met):**

- ✅ ``utils/metrics.py`` degrades to no-ops without the extra; ``/metrics``
  501s rather than erroring.
- ✅ 9 new tests green (6 unit on ``record_pipeline_run`` + 3 on the endpoint).
- ✅ ``record_pipeline_run`` hooked into all four pipeline terminal paths.
- ✅ Dashboard JSON + all provisioning YAML parse; ``docker-compose`` overlay
  validated.
- ✅ Ruff clean; existing api + graph tests unaffected (2 unrelated WinError
  10055 socket-exhaustion failures reproduce only in the long combined run and
  pass in isolation).

---

## ADR-032: Security & Reliability Hardening (auth fail-closed, OCR off-loop, scheduled audit verify, frontend headers)

**Date:** 2026-05-29
**Status:** Accepted

**Context:**
A post-launch review (and a deep external audit) surfaced four latent issues
spanning both repos:

1. **Silent unsigned-token acceptance.** With no ``SAR_JWT_SECRET`` set,
   ``verify_token`` fell back to a base64(json(UserContext)) bearer shape and
   logged a warning — but still *accepted* it. That proves no identity; any
   caller could impersonate any user. The HF Space sets no secret (it uses the
   BYOK persona contract, not bearer auth), so the REST ``/query`` / ``/audit``
   endpoints were effectively unauthenticated there.
2. **Event-loop blocking in ingestion.** ``ingest_document`` is ``async`` but
   called the synchronous ``_apply_ocr_fallback`` inline. PaddleOCR is
   CPU-bound and the VLM fallback's sync wrappers spin their own loop, so a
   scanned PDF stalled the whole event loop. (The audit blamed
   ``ingestion/contextual.py``; that was a misdiagnosis — contextual.py is
   already bounded-async via ``asyncio.gather`` + a semaphore.)
3. **Tamper-evidence nobody checked.** The SHA-256 audit chain was verifiable
   on demand but never automatically. Worse, ``schedule_session_purge`` was
   defined yet never wired — the FastAPI app had no ``lifespan`` — so BYOK
   session collections never auto-purged.
4. **No HTTP security headers** on the Vercel frontend (no CSP, no
   clickjacking / MIME-sniffing / referrer protections).

**Decision:**

- **Auth fails closed.** New ``allow_unsigned_tokens`` setting (default
  ``False``). With no secret and the flag off, every bearer token is rejected.
  The legacy unsigned shape is honoured only when
  ``SAR_ALLOW_UNSIGNED_TOKENS=true`` (dev/test). ``mint_dev_token`` mirrors the
  policy and raises rather than emit an unsigned token. The test suite opts in
  via an autouse conftest fixture; a new test pins the fail-closed default.
- **OCR off the loop.** ``ingest_document`` wraps the OCR pass in
  ``asyncio.to_thread(self._apply_ocr_fallback, ...)`` — the correct, narrow
  fix at the real blocking site.
- **Scheduled audit verification.** New ``utils/audit_verify`` runs
  ``AuditLogger.verify_chain()`` once at boot and every
  ``SAR_AUDIT_VERIFY_INTERVAL_HOURS`` (default 6) from a FastAPI ``lifespan``,
  surfacing a broken chain via structured error log + Prometheus metrics
  (``audit_chain_verifications_total{result}``, ``audit_chain_valid`` gauge).
  The same lifespan finally wires ``schedule_session_purge`` (gated on
  ``byok_mode``). Both jobs are best-effort — a scheduler failure never blocks
  startup — and degrade to a single startup sweep without APScheduler.
- **Frontend security headers** via ``next.config.ts`` ``headers()``: a CSP
  (same-origin + ``*.hf.space`` + Vercel analytics; no ``unsafe-eval``),
  ``X-Frame-Options: DENY``, ``X-Content-Type-Options: nosniff``,
  ``Referrer-Policy``, ``Permissions-Policy``, and HSTS on every route.

**Consequences:**

- (+) The REST surface can no longer be impersonated without a configured
  signing key — fail closed by default, opt-in for dev only.
- (+) Concurrent ingests and chat stay responsive during OCR; the API event
  loop is no longer monopolised by a scanned PDF.
- (+) Audit tampering is detected automatically and exported as a metric the
  Grafana dashboard can alert on; BYOK session collections finally auto-purge.
- (+) The frontend is hardened against clickjacking, MIME sniffing, and
  referrer leakage; verified live on ``secureagentrag-web.vercel.app``.
- (−) Smoke scripts that minted unsigned tokens must now set
  ``SAR_ALLOW_UNSIGNED_TOKENS=true`` (or a real secret) against a live server.
- (−) The CSP grants ``'unsafe-inline'`` for scripts/styles (no nonce pipeline
  in this Next setup) — weaker than a nonce-based policy, acceptable for the
  demo; ``'unsafe-eval'`` is not granted.

**Acceptance criteria (all met):**

- ✅ New fail-closed auth test + updated legacy tests green; full auth + api
  suites pass.
- ✅ OCR wrapped in ``to_thread``; pipeline tests green.
- ✅ ``utils/audit_verify`` with 5 tests; lifespan runs clean under TestClient.
- ✅ All six security headers verified live (local prod build + Vercel deploy);
  ``/`` and ``/chat`` still 200.
- ✅ Backend CI green; Vercel build + lint green.

