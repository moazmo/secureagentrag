"""Tests for cloud LLM provider clients (Groq, OpenAI, Anthropic)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from inference.cloud_clients import (
    AnthropicClient,
    GroqClient,
    LLMProvider,
    OpenAIClient,
)
from inference.llm_factory import LLMResponse


class TestLLMProvider:
    """Tests for the LLMProvider enum."""

    def test_provider_values(self) -> None:
        """LLMProvider enum should have expected values."""
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.GROQ == "groq"
        assert LLMProvider.OPENAI == "openai"
        assert LLMProvider.ANTHROPIC == "anthropic"


class TestGroqClient:
    """Tests for the GroqClient."""

    @pytest.fixture()
    def client(self) -> GroqClient:
        """Create a Groq client with test API key."""
        return GroqClient(api_key="test-groq-key", model="llama-3.3-70b-versatile")

    @pytest.fixture()
    def mock_openai_response(self) -> dict:
        """Sample OpenAI-compatible response from Groq."""
        return {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "model": "llama-3.3-70b-versatile",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from Groq!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 10,
                "total_tokens": 30,
            },
        }

    @pytest.mark.asyncio
    async def test_generate_success(self, client: GroqClient, mock_openai_response: dict) -> None:
        """Generate should return LLMResponse from Groq API."""
        mock_response = httpx.Response(
            status_code=200,
            json=mock_openai_response,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.generate(prompt="Hello", system_prompt="Be concise")

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello from Groq!"
        assert result.provider == "groq"
        assert result.model == "llama-3.3-70b-versatile"
        assert result.usage["prompt_tokens"] == 20
        assert result.usage["completion_tokens"] == 10
        assert result.usage["total_tokens"] == 30

    @pytest.mark.asyncio
    async def test_chat_success(self, client: GroqClient, mock_openai_response: dict) -> None:
        """Chat should send messages and return LLMResponse."""
        mock_response = httpx.Response(
            status_code=200,
            json=mock_openai_response,
            request=httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            messages = [{"role": "user", "content": "Hello"}]
            result = await client.chat(messages=messages)

        assert result.text == "Hello from Groq!"
        assert result.provider == "groq"

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: GroqClient) -> None:
        """Health check should return True on 200."""
        mock_response = httpx.Response(
            status_code=200,
            json={"data": []},
            request=httpx.Request("GET", "https://api.groq.com/openai/v1/models"),
        )

        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_connection_failure(self, client: GroqClient) -> None:
        """Health check should return False on connection failure."""
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("unreachable")
            result = await client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_error_status_raises(self, client: GroqClient) -> None:
        """Should raise after retrying on a 429 rate-limit response.

        The cloud client now retries 429 with exponential backoff and re-
        raises ``_RateLimitError`` (an internal sentinel) once attempts are
        exhausted. We assert *either* signal so the test matches both the
        pre-retry HTTPStatusError contract and the new rate-limit handling.
        """
        from inference.cloud_clients import _RateLimitError

        mock_response = httpx.Response(
            status_code=429,
            json={"error": {"message": "Rate limit exceeded"}},
            request=httpx.Request(
                "POST", "https://api.groq.com/openai/v1/chat/completions"
            ),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises((httpx.HTTPStatusError, _RateLimitError)):
                await client.generate(prompt="test")


class TestOpenAIClient:
    """Tests for the OpenAIClient."""

    @pytest.fixture()
    def client(self) -> OpenAIClient:
        """Create an OpenAI client with test API key."""
        return OpenAIClient(api_key="test-openai-key", model="gpt-4o-mini")

    @pytest.fixture()
    def mock_openai_response(self) -> dict:
        """Sample OpenAI chat completion response."""
        return {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "model": "gpt-4o-mini",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello from OpenAI!"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 8,
                "total_tokens": 23,
            },
        }

    @pytest.mark.asyncio
    async def test_generate_success(self, client: OpenAIClient, mock_openai_response: dict) -> None:
        """Generate should return LLMResponse from OpenAI API."""
        mock_response = httpx.Response(
            status_code=200,
            json=mock_openai_response,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.generate(prompt="Hello", system_prompt="Be helpful")

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello from OpenAI!"
        assert result.provider == "openai"
        assert result.model == "gpt-4o-mini"
        assert result.usage["total_tokens"] == 23

    @pytest.mark.asyncio
    async def test_chat_success(self, client: OpenAIClient, mock_openai_response: dict) -> None:
        """Chat should send messages and return LLMResponse."""
        mock_response = httpx.Response(
            status_code=200,
            json=mock_openai_response,
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            messages = [
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": "What is AI?"},
            ]
            result = await client.chat(messages=messages)

        assert result.text == "Hello from OpenAI!"
        assert result.provider == "openai"

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: OpenAIClient) -> None:
        """Health check should return True on 200."""
        mock_response = httpx.Response(
            status_code=200,
            json={"data": []},
            request=httpx.Request("GET", "https://api.openai.com/v1/models"),
        )

        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_error_status_raises(self, client: OpenAIClient) -> None:
        """Should raise on HTTP error status."""
        mock_response = httpx.Response(
            status_code=500,
            json={"error": {"message": "Internal server error"}},
            request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(httpx.HTTPStatusError):
                await client.generate(prompt="test")


class TestAnthropicClient:
    """Tests for the AnthropicClient."""

    @pytest.fixture()
    def client(self) -> AnthropicClient:
        """Create an Anthropic client with test API key."""
        return AnthropicClient(api_key="test-anthropic-key", model="claude-sonnet-4-20250514")

    @pytest.fixture()
    def mock_anthropic_response(self) -> dict:
        """Sample Anthropic Messages API response."""
        return {
            "id": "msg_789",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello from Claude!"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 12,
                "output_tokens": 5,
            },
        }

    @pytest.mark.asyncio
    async def test_generate_success(
        self, client: AnthropicClient, mock_anthropic_response: dict
    ) -> None:
        """Generate should return LLMResponse from Anthropic API."""
        mock_response = httpx.Response(
            status_code=200,
            json=mock_anthropic_response,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.generate(prompt="Hello", system_prompt="Be helpful")

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello from Claude!"
        assert result.provider == "anthropic"
        assert result.model == "claude-sonnet-4-20250514"
        assert result.usage["prompt_tokens"] == 12
        assert result.usage["completion_tokens"] == 5
        assert result.usage["total_tokens"] == 17

    @pytest.mark.asyncio
    async def test_chat_with_system_message(
        self, client: AnthropicClient, mock_anthropic_response: dict
    ) -> None:
        """Chat should extract system message and send properly formatted request."""
        mock_response = httpx.Response(
            status_code=200,
            json=mock_anthropic_response,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            messages = [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "Write hello world"},
            ]
            result = await client.chat(messages=messages)

        assert result.text == "Hello from Claude!"
        # Verify the system message was extracted and sent as top-level param
        call_kwargs = mock_post.call_args
        payload = call_kwargs.kwargs.get("json", call_kwargs[1].get("json", {}))
        assert payload.get("system") == "You are a coding assistant."
        # Messages should not contain the system message
        assert all(m["role"] != "system" for m in payload["messages"])

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: AnthropicClient) -> None:
        """Health check should return True when API responds."""
        mock_response = httpx.Response(
            status_code=200,
            json={
                "id": "msg_test",
                "content": [{"type": "text", "text": "hi"}],
                "model": "claude-sonnet-4-20250514",
                "usage": {"input_tokens": 1, "output_tokens": 1},
            },
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_connection_failure(self, client: AnthropicClient) -> None:
        """Health check should return False on connection failure."""
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("unreachable")
            result = await client.health_check()

        assert result is False

    @pytest.mark.asyncio
    async def test_error_status_raises(self, client: AnthropicClient) -> None:
        """Should raise on HTTP error status."""
        mock_response = httpx.Response(
            status_code=401,
            json={"error": {"type": "authentication_error", "message": "Invalid key"}},
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            with pytest.raises(httpx.HTTPStatusError):
                await client.generate(prompt="test")
