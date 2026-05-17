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

    # ── Chunking ─────────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # ── Retrieval ────────────────────────────────────────────────────────────────
    top_k: int = 10
    rerank_top_k: int = 5
    relevance_threshold: float = 0.7

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

    # ── PostgreSQL (for LangGraph checkpointing) ─────────────────────────────────
    postgres_url: str = "postgresql://sar_user:sar_password@localhost:5432/secureagentrag"

    # ── Redis (for distributed rate limiting / caching) ──────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    use_redis_rate_limiter: bool = False


# Singleton instance — import this throughout the application
settings = Settings()
