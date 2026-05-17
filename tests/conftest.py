"""Shared pytest fixtures for SecureAgentRAG test suite."""

from __future__ import annotations

from typing import Any

import pytest

from config.settings import Settings


@pytest.fixture()
def test_settings() -> Settings:
    """Create a Settings instance with test-specific overrides.

    Uses in-memory / embedded configurations to avoid requiring
    external services during unit tests.
    """
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        debug=True,
        log_level="DEBUG",
        qdrant_url="http://localhost:6333",
        qdrant_collection="test_documents",
        ollama_url="http://localhost:11434",
        llm_model="qwen3:8b",
        embedding_model="bge-m3",
        embedding_dim=1024,
        enable_rbac=False,
        phoenix_endpoint=None,
    )


@pytest.fixture()
def sample_text() -> dict[str, str]:
    """Provide sample text in English and Arabic for testing parsers and chunkers."""
    return {
        "english": (
            "Retrieval-Augmented Generation (RAG) is a technique that combines "
            "information retrieval with language model generation. It first retrieves "
            "relevant documents from a knowledge base, then uses those documents as "
            "context for generating accurate and grounded responses. This approach "
            "significantly reduces hallucinations and improves factual accuracy."
        ),
        "arabic": (
            "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهتم بإنشاء أنظمة قادرة على "
            "أداء مهام تتطلب عادةً ذكاءً بشرياً. يشمل ذلك التعلم الآلي ومعالجة اللغة "
            "الطبيعية والرؤية الحاسوبية والروبوتات."
        ),
    }


@pytest.fixture()
def sample_metadata() -> dict[str, Any]:
    """Provide sample document metadata for testing ingestion and retrieval."""
    return {
        "source": "test_document.pdf",
        "page_number": 1,
        "total_pages": 10,
        "department": "engineering",
        "access_roles": ["admin", "engineer"],
        "language": "en",
        "ingested_by": "test_user",
    }
