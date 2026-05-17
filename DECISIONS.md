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
