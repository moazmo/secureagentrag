"""Tests for the LLM factory module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from inference.cloud_clients import AnthropicClient, GroqClient, OpenAIClient
from inference.llm_factory import LLMResponse, chat, clear_llm_cache, generate, get_llm
from inference.ollama_client import OllamaClient


class TestLLMResponse:
    """Tests for the LLMResponse model."""

    def test_create_minimal(self) -> None:
        """LLMResponse should be creatable with required fields only."""
        response = LLMResponse(text="Hello", model="test", provider="ollama")
        assert response.text == "Hello"
        assert response.model == "test"
        assert response.provider == "ollama"
        assert response.usage == {}
        assert response.latency_ms == 0.0
        assert response.metadata == {}

    def test_create_full(self) -> None:
        """LLMResponse should accept all optional fields."""
        response = LLMResponse(
            text="Generated text",
            model="gpt-4o-mini",
            provider="openai",
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            latency_ms=150.5,
            metadata={"finish_reason": "stop"},
        )
        assert response.usage["total_tokens"] == 30
        assert response.latency_ms == 150.5
        assert response.metadata["finish_reason"] == "stop"

    def test_serialization(self) -> None:
        """LLMResponse should serialize to dict properly."""
        response = LLMResponse(text="test", model="m", provider="p")
        data = response.model_dump()
        assert data["text"] == "test"
        assert "usage" in data
        assert "latency_ms" in data


class TestGetLLM:
    """Tests for the get_llm factory function."""

    def test_get_ollama_client(self) -> None:
        """get_llm('ollama') should return an OllamaClient."""
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.ollama_url = "http://localhost:11434"
            client = get_llm("ollama")
        assert isinstance(client, OllamaClient)

    def test_get_groq_client_with_key(self) -> None:
        """get_llm('groq') with API key should return GroqClient."""
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.groq_api_key = "test-key"
            client = get_llm("groq")
        assert isinstance(client, GroqClient)
        assert client.model == "llama-3.3-70b-versatile"

    def test_get_groq_client_without_key_raises(self) -> None:
        """get_llm('groq') without API key should raise ValueError."""
        clear_llm_cache()
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.groq_api_key = None
            with pytest.raises(ValueError, match="Groq API key not configured"):
                get_llm("groq")

    def test_get_openai_client_with_key(self) -> None:
        """get_llm('openai') with API key should return OpenAIClient."""
        clear_llm_cache()
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.openai_api_key = "test-key"
            client = get_llm("openai")
        assert isinstance(client, OpenAIClient)
        assert client.model == "gpt-4o-mini"

    def test_get_openai_client_without_key_raises(self) -> None:
        """get_llm('openai') without API key should raise ValueError."""
        clear_llm_cache()
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.openai_api_key = None
            with pytest.raises(ValueError, match="OpenAI API key not configured"):
                get_llm("openai")

    def test_get_anthropic_client_with_key(self) -> None:
        """get_llm('anthropic') with API key should return AnthropicClient."""
        clear_llm_cache()
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_api_key = "test-key"
            client = get_llm("anthropic")
        assert isinstance(client, AnthropicClient)
        assert client.model == "claude-sonnet-4-20250514"

    def test_get_anthropic_client_without_key_raises(self) -> None:
        """get_llm('anthropic') without API key should raise ValueError."""
        clear_llm_cache()
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.anthropic_api_key = None
            with pytest.raises(ValueError, match="Anthropic API key not configured"):
                get_llm("anthropic")

    def test_unknown_provider_raises(self) -> None:
        """get_llm with unknown provider should raise ValueError."""
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            with pytest.raises(ValueError, match="Unknown LLM provider"):
                get_llm("unknown_provider")

    def test_default_provider_from_settings(self) -> None:
        """get_llm() without provider should use settings.default_provider."""
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.ollama_url = "http://localhost:11434"
            client = get_llm()
        assert isinstance(client, OllamaClient)

    def test_custom_model_override(self) -> None:
        """get_llm with model param should override default."""
        with patch("inference.llm_factory.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.ollama_url = "http://localhost:11434"
            client = get_llm("ollama", model="llama3:latest")
        assert client.model == "llama3:latest"


class TestGenerateConvenience:
    """Tests for the generate() convenience function."""

    @pytest.mark.asyncio
    async def test_generate_does_not_close_cached_client(self) -> None:
        """generate() must NOT close the client — get_llm returns a cached,
        shared instance, and closing it would break the next caller."""
        mock_response = LLMResponse(text="Hello!", model="test-model", provider="ollama")
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client.close = AsyncMock()

        with patch("inference.llm_factory.get_llm", return_value=mock_client):
            result = await generate(prompt="Hi", provider="ollama")

        assert result.text == "Hello!"
        assert result.latency_ms > 0
        mock_client.close.assert_not_called()


class TestClientCaching:
    """Tests for LLM client caching behavior."""

    def test_get_llm_caches_clients(self) -> None:
        """get_llm should return cached client for same provider+model."""
        clear_llm_cache()
        with patch("inference.ollama_client.OllamaClient") as mock_cls:
            client1 = get_llm(provider="ollama", model="qwen3:8b")
            client2 = get_llm(provider="ollama", model="qwen3:8b")
            # Should only create one instance
            mock_cls.assert_called_once()
            assert client1 is client2

    def test_get_llm_different_models_create_separate_clients(self) -> None:
        """Different models should get separate cached clients."""
        clear_llm_cache()
        with patch("inference.ollama_client.OllamaClient") as mock_cls:
            get_llm(provider="ollama", model="model-a")
            get_llm(provider="ollama", model="model-b")
            assert mock_cls.call_count == 2

    def test_clear_llm_cache_clears_all(self) -> None:
        """clear_llm_cache should empty the cache."""
        clear_llm_cache()
        with patch("inference.ollama_client.OllamaClient") as mock_cls:
            get_llm(provider="ollama", model="test")
            clear_llm_cache()
            get_llm(provider="ollama", model="test")
            assert mock_cls.call_count == 2


class TestChatConvenience:
    """Tests for the chat() convenience function."""

    @pytest.mark.asyncio
    async def test_chat_creates_and_closes_client(self) -> None:
        """chat() should create client, call chat, and close."""
        mock_response = LLMResponse(text="I can help!", model="test-model", provider="ollama")
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.close = AsyncMock()

        with patch("inference.llm_factory.get_llm", return_value=mock_client):
            messages = [{"role": "user", "content": "Help me"}]
            result = await chat(messages=messages, provider="ollama")

        assert result.text == "I can help!"
        assert result.latency_ms > 0
        # With client caching, close is no longer called per-request
        # Clients are reused across requests for connection pooling
