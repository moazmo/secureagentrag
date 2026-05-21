"""Application settings managed via pydantic-settings with environment variable support."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for SecureAgentRAG.

    All settings can be overridden via environment variables prefixed with ``SAR_``.
    For example, ``SAR_DEBUG=true`` sets ``debug`` to True.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SAR_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────────
    app_name: str = "SecureAgentRAG"
    debug: bool = False
    log_level: str = "INFO"

    # ── Qdrant Vector Store ──────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    qdrant_api_key: str | None = None

    # ── Ollama / LLM ─────────────────────────────────────────────────────────────
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:8b"
    embedding_model: str = "bge-m3"
    embedding_dim: int = 1024
    embedding_backend: str = "ollama"  # "ollama" or "local" (sentence-transformers)
    local_embedding_model: str = "BAAI/bge-m3"
    # How long Ollama keeps models resident in VRAM between requests.
    # On consumer hardware the LLM (qwen3:8b ~5.5GB) and embedding (bge-m3 ~1.2GB)
    # need to swap if VRAM is tight. Long keep-alive avoids ~5-10s reload per swap.
    ollama_keep_alive: str = "30m"

    # ── Chunking ─────────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ────────────────────────────────────────────────────────────────
    top_k: int = 10
    rerank_top_k: int = 5
    relevance_threshold: float = 0.7
    # RAG Fusion: generate N query reformulations, retrieve in parallel,
    # fuse the ranked lists via RRF. Boosts recall on under-specified
    # queries. Cost: N-1 extra LLM calls + N parallel Qdrant searches.
    # Set to 1 to disable.
    rag_fusion_n_queries: int = 3
    rag_fusion_enabled: bool = True
    # ── Reranker ─────────────────────────────────────────────────────────────────
    # Re-score retrieved documents for higher precision.
    # Options: "none" (disabled), "cross_encoder" (BGE-Reranker-v2-M3),
    # "colbert" (ColBERTv2 late-interaction, requires colbert-ai package).
    # The cross-encoder downloads ~600MB from HuggingFace on first use.
    # The ColBERT checkpoint is ~400MB. Disabled by default so the first
    # query does not silently hang on download. Pre-download explicitly.
    reranker_type: str = "none"
    reranker_checkpoint: str = "BAAI/bge-reranker-v2-m3"
    colbert_checkpoint: str = "colbert-ir/colbertv2.0"
    # Path to a locally fine-tuned cross-encoder checkpoint produced by
    # scripts/train_reranker.py. Used when reranker_type == "fine_tuned".
    finetuned_reranker_path: str = "data/checkpoints/reranker-domain-v1"

    # ── Inference Providers ──────────────────────────────────────────────────────
    default_provider: str = "ollama"
    cloud_provider: str | None = None
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    groq_api_base: str = "https://api.groq.com/openai/v1"
    openai_api_base: str = "https://api.openai.com/v1"
    anthropic_api_base: str = "https://api.anthropic.com/v1"

    # ── RAG Pipeline Thresholds ───────────────────────────────────────────────────
    relevance_retry_threshold: float = 0.5
    confidence_threshold: float = 0.6
    max_retries: int = 2

    # ── JSON Citations ────────────────────────────────────────────────────────────
    # When enabled, the synthesizer requests structured JSON output from the LLM
    # with `answer` and `citations` fields instead of relying on regex extraction.
    json_citations_enabled: bool = False

    # ── Embedding Batch Size ──────────────────────────────────────────────────────
    embedding_batch_size: int = 32  # Max texts per embedding API call
    embedding_max_concurrent_batches: int = 4  # Max concurrent batch requests

    # ── RBAC ─────────────────────────────────────────────────────────────────────
    enable_rbac: bool = True

    # ── Observability (Phoenix) ──────────────────────────────────────────────────
    phoenix_endpoint: str | None = None

    # ── Sparse Vectors (Qdrant native, replaces rank_bm25 pickle) ────────────────
    sparse_backend: str = "bm25"  # "bm25" | "splade"
    sparse_vector_name: str = "sparse"
    sparse_model: str = "naver/splade-cocondenser-ensembledistil"

    # ── Audit + Conversation Storage ──────────────────────────────────────────────
    audit_log_dir: str = "audit_logs"
    conversation_dir: str = "conversations"
    checkpoint_db_path: str = "data/checkpoints.sqlite"
    # Opt-in: enable persistent (SQLite/Postgres) LangGraph checkpointing.
    # Default off because pytest-asyncio creates per-test event loops which
    # collide with aiosqlite's loop-bound connection. For production single-
    # process Streamlit / FastAPI deployments, set SAR_USE_PERSISTENT_CHECKPOINTER=true.
    use_persistent_checkpointer: bool = False

    # ── PostgreSQL (for LangGraph checkpointing) ─────────────────────────────────
    postgres_url: str = "postgresql://sar_user:sar_password@localhost:5433/secureagentrag"

    # ── Pipeline SLO ─────────────────────────────────────────────────────────────
    # Hard wall-clock budget for a single RAG pipeline run (rewrite loop +
    # retrieval + grading + synthesis + evaluation). On timeout the caller
    # gets a graceful refusal + audit entry; nothing partial is rendered as
    # if the answer succeeded. 0 disables the deadline.
    request_timeout_s: float = 60.0

    # ── Authentication ───────────────────────────────────────────────────────────
    # When ``jwt_secret`` is set the FastAPI / MCP layers verify HS256-signed
    # JWTs and derive UserContext from validated claims. When unset, callers
    # fall back to the dev-mode base64(json(UserContext)) token shape so
    # existing tests and smoke scripts keep working — but a runtime warning is
    # emitted on every request. Production deployments MUST set this.
    #
    # ``jwt_issuer`` / ``jwt_audience`` are checked against ``iss`` / ``aud``
    # claims when present. Leave empty to disable that check (default).
    # ``jwt_ttl_seconds`` is the lifetime of tokens minted via the local
    # ``/token`` dev endpoint; real IdPs (Keycloak/Auth0) set their own.
    jwt_secret: str | None = None
    jwt_issuer: str = "secureagentrag"
    jwt_audience: str = "secureagentrag-api"
    jwt_ttl_seconds: int = 3600
    jwt_algorithm: str = "HS256"
    # JWKS endpoint for RS256 verification (e.g. Keycloak, Auth0).
    # When set and jwt_algorithm == "RS256", tokens are verified against
    # the cached JWKS instead of jwt_secret.
    jwks_url: str | None = None
    jwks_cache_ttl_seconds: int = 300

    # ── Citation Faithfulness Gate (NLI) ─────────────────────────────────────────
    # After synthesis, run a per-sentence NLI check: for each sentence that
    # carries an inline `[N]` citation, ask a yes/no entailment question
    # against the cited chunk's text. Sentences that fail are either marked
    # `[unsupported]` (soft mode) or dropped from the answer (strict mode).
    # The check uses the same local LLM as the rest of the graph — no extra
    # model download. Cost: one LLM call per cited sentence (parallel).
    faithfulness_gate_enabled: bool = False
    faithfulness_gate_mode: str = "flag"  # "flag" | "drop"
    faithfulness_threshold: float = 0.7  # min entailment ratio to consider answer faithful
    faithfulness_max_concurrent: int = 4  # parallel NLI checks

    # ── Redis (for distributed rate limiting / caching) ──────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    use_redis_rate_limiter: bool = False

    # ── PII Redaction ────────────────────────────────────────────────────────────
    # Scrub email, phone, SSN, credit-card, IBAN, IP address before persisting
    # to audit log / query cache. Defense against accidental PII leakage into
    # secondary stores. Regex-based by default; if Microsoft Presidio is
    # installed it is used automatically for higher recall.
    pii_redaction_enabled: bool = True

    # ── Prompt-Injection Guardrails ──────────────────────────────────────────────
    # Run a regex + heuristic check on the user query before retrieval. Blocks
    # obvious jailbreak / system-prompt-override attempts. Logged via the audit
    # logger as ``security_block`` events.
    guardrails_enabled: bool = True
    # Strict mode: after the fast regex gate, escalate ambiguous or all queries
    # to a local LLM-based classifier for a second opinion. Adds one LLM call
    # per query but catches adversarial inputs that evade regex patterns.
    guardrails_strict: bool = False
    # Escalation backend used in strict mode. Options:
    #   "llm"        — legacy SAFE/UNSAFE prompt on the synth-grade model
    #                  (core.agents.guardrails_llm). Default for backward
    #                  compatibility.
    #   "llamaguard" — Meta's LlamaGuard 3 8B via Ollama. Use with
    #                  ``ollama pull llama-guard3:8b``. More accurate on
    #                  the standard S1-S14 taxonomy.
    guardrails_backend: str = "llm"
    llamaguard_model: str = "llama-guard3:8b"

    # ── Contextual Retrieval (Anthropic 2024 technique) ──────────────────────────
    # Prepend a short LLM-generated context summary to each chunk before
    # embedding. Adds 1 cheap LLM call per chunk at ingestion time but
    # measurably improves retrieval recall (Anthropic reported ~35-49%
    # failure reduction). Local Qwen3-8B is fine for the summary.
    contextual_retrieval_enabled: bool = False

    # ── VLM OCR (Primary OCR via vision-language model) ───────────────────────────
    # Use a VLM (Qwen2.5-VL / Qwen3-VL, LLaVA, etc.) via Ollama as the primary OCR path.
    # Superior to PaddleOCR on complex layouts, tables, and mixed-language
    # documents. Falls back to PaddleOCR when the VLM is unavailable.
    vlm_ocr_enabled: bool = False
    vlm_ocr_model: str = "qwen2.5-vl"

    # ── Multi-Tenancy ────────────────────────────────────────────────────────────
    # When true, each organization gets its own Qdrant collection
    # (documents_{org_id}). This provides stronger isolation than payload-level
    # RBAC filtering but requires creating collections per org on first use.
    # When false, all docs share a single collection with RBAC at payload level.
    multi_tenant_collections: bool = False

    # ── Multi-Modal RAG ──────────────────────────────────────────────────────────
    # When ingesting images, also generate a rich text description using a VLM.
    # The description is embedded as a separate chunk, enabling retrieval for
    # queries like "what does the diagram show?" without requiring CLIP or
    # other multi-modal embedding models.
    multimodal_descriptions_enabled: bool = False

    # ── Self-Query Retrieval ─────────────────────────────────────────────────────
    # Extract structured metadata filters (source_file, date_range,
    # sensitivity_level, roles) from the natural language query using a small
    # local LLM prompt. The filters are merged with the RBAC filter and passed
    # to Qdrant, scoping retrieval before embedding search runs.
    self_query_enabled: bool = False

    # ── HyDE (Hypothetical Document Embeddings) ──────────────────────────────────
    # Generate a hypothetical answer to the query, embed *that* instead of the
    # raw query. Boosts recall when query vocabulary differs from doc
    # vocabulary (questions vs declarative sentences). Adds one LLM call per
    # query — skip for simple keyword lookups; enable for complex questions.
    hyde_enabled: bool = False

    # ── Pricing for cost dashboard (USD per 1M tokens) ───────────────────────────
    # Used by evaluation/cost.py to convert recorded usage into $/query.
    price_groq_input_per_1m: float = 0.59
    price_groq_output_per_1m: float = 0.79
    price_openai_input_per_1m: float = 2.50
    price_openai_output_per_1m: float = 10.00
    price_anthropic_input_per_1m: float = 3.00
    price_anthropic_output_per_1m: float = 15.00
    # Local inference: estimated electricity cost only (consumer hardware).
    # 200W GPU @ $0.15/kWh ≈ $0.03/hour ≈ $0.000008/sec
    price_local_per_second: float = 0.000008


# Singleton instance — import this throughout the application
settings = Settings()
