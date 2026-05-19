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
    # The reranker (BAAI/bge-reranker-v2-m3 cross-encoder) downloads ~600MB
    # from HuggingFace the first time it is used. Disabled by default so the
    # first query does not silently hang on the download. Enable explicitly
    # after pre-downloading the model (uv run python -c
    # "from sentence_transformers import CrossEncoder;
    #  CrossEncoder('BAAI/bge-reranker-v2-m3')").
    enable_reranker: bool = False

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

    # ── Embedding Batch Size ──────────────────────────────────────────────────────
    embedding_batch_size: int = 32  # Max texts per embedding API call
    embedding_max_concurrent_batches: int = 4  # Max concurrent batch requests

    # ── RBAC ─────────────────────────────────────────────────────────────────────
    enable_rbac: bool = True

    # ── Observability (Phoenix) ──────────────────────────────────────────────────
    phoenix_endpoint: str | None = None

    # ── BM25 Persistence ──────────────────────────────────────────────────────────
    bm25_index_path: str = "data/bm25_index.pkl"

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
    postgres_url: str = "postgresql://sar_user:sar_password@localhost:5432/secureagentrag"

    # ── Redis (for distributed rate limiting / caching) ──────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    use_redis_rate_limiter: bool = False


# Singleton instance — import this throughout the application
settings = Settings()
