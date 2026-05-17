"""Tests for the Ollama client wrapper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from inference.llm_factory import LLMResponse
from inference.ollama_client import OllamaClient


class TestOllamaClientInit:
    """Tests for OllamaClient initialization."""

    def test_default_initialization(self) -> None:
        """Client should use settings defaults when no params provided."""
        with patch("inference.ollama_client.settings") as mock_settings:
            mock_settings.ollama_url = "http://localhost:11434"
            mock_settings.llm_model = "qwen3:8b"
            client = OllamaClient()
            assert client.base_url == "http://localhost:11434"
            assert client.model == "qwen3:8b"
            assert client.timeout == 120.0

    def test_custom_initialization(self) -> None:
        """Client should use custom params when provided."""
        client = OllamaClient(
            base_url="http://custom:9999",
            model="llama3:latest",
            timeout=60.0,
        )
        assert client.base_url == "http://custom:9999"
        assert client.model == "llama3:latest"
        assert client.timeout == 60.0

    def test_trailing_slash_stripped(self) -> None:
        """Client should strip trailing slash from base_url."""
        client = OllamaClient(base_url="http://localhost:11434/")
        assert client.base_url == "http://localhost:11434"


class TestOllamaClientGenerate:
    """Tests for OllamaClient.generate()."""

    @pytest.fixture()
    def client(self) -> OllamaClient:
        """Create a client with known parameters."""
        return OllamaClient(base_url="http://localhost:11434", model="test-model")

    @pytest.fixture()
    def mock_generate_response(self) -> dict:
        """Sample Ollama /api/generate response."""
        return {
            "model": "test-model",
            "response": "Hello! I am an AI assistant.",
            "done": True,
            "prompt_eval_count": 10,
            "eval_count": 8,
            "total_duration": 500000000,
            "load_duration": 100000000,
        }

    @pytest.mark.asyncio
    async def test_generate_success(
        self, client: OllamaClient, mock_generate_response: dict
    ) -> None:
        """Generate should return LLMResponse on successful API call."""
        mock_response = httpx.Response(
            status_code=200,
            json=mock_generate_response,
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            result = await client.generate(prompt="Hello", system_prompt="Be helpful")

        assert isinstance(result, LLMResponse)
        assert result.text == "Hello! I am an AI assistant."
        assert result.model == "test-model"
        assert result.provider == "ollama"
        assert result.usage["prompt_tokens"] == 10
        assert result.usage["completion_tokens"] == 8
        assert result.usage["total_tokens"] == 18
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_generate_sends_correct_payload(self, client: OllamaClient) -> None:
        """Generate should send proper payload to Ollama API."""
        mock_response = httpx.Response(
            status_code=200,
            json={"model": "test-model", "response": "ok", "done": True},
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            await client.generate(
                prompt="test prompt",
                system_prompt="system",
                temperature=0.5,
                max_tokens=1024,
            )

        call_kwargs = mock_post.call_args
        payload = (
            call_kwargs.kwargs["json"] if "json" in call_kwargs.kwargs else call_kwargs[1]["json"]
        )
        assert payload["model"] == "test-model"
        assert payload["prompt"] == "test prompt"
        assert payload["system"] == "system"
        assert payload["stream"] is False
        assert payload["options"]["temperature"] == 0.5
        assert payload["options"]["num_predict"] == 1024


class TestOllamaClientChat:
    """Tests for OllamaClient.chat()."""

    @pytest.fixture()
    def client(self) -> OllamaClient:
        """Create a client with known parameters."""
        return OllamaClient(base_url="http://localhost:11434", model="test-model")

    @pytest.mark.asyncio
    async def test_chat_success(self, client: OllamaClient) -> None:
        """Chat should return LLMResponse from Ollama chat API."""
        mock_data = {
            "model": "test-model",
            "message": {"role": "assistant", "content": "I can help with that."},
            "done": True,
            "prompt_eval_count": 15,
            "eval_count": 6,
            "total_duration": 400000000,
            "load_duration": 50000000,
        }
        mock_response = httpx.Response(
            status_code=200,
            json=mock_data,
            request=httpx.Request("POST", "http://localhost:11434/api/chat"),
        )

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            messages = [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "What is RAG?"},
            ]
            result = await client.chat(messages=messages)

        assert isinstance(result, LLMResponse)
        assert result.text == "I can help with that."
        assert result.provider == "ollama"
        assert result.usage["prompt_tokens"] == 15
        assert result.usage["completion_tokens"] == 6


class TestOllamaClientHealthCheck:
    """Tests for OllamaClient.health_check()."""

    @pytest.fixture()
    def client(self) -> OllamaClient:
        """Create a client with known parameters."""
        return OllamaClient(base_url="http://localhost:11434", model="test-model")

    @pytest.mark.asyncio
    async def test_health_check_success(self, client: OllamaClient) -> None:
        """Health check should return True when server responds 200."""
        mock_response = httpx.Response(
            status_code=200,
            json={"models": []},
            request=httpx.Request("GET", "http://localhost:11434/api/tags"),
        )

        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = mock_response
            result = await client.health_check()

        assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client: OllamaClient) -> None:
        """Health check should return False when server is unreachable."""
        with patch.object(client._client, "get", new_callable=AsyncMock) as mock_get:
            mock_get.side_effect = httpx.ConnectError("Connection refused")
            result = await client.health_check()

        assert result is False


class TestOllamaClientStream:
    """Tests for OllamaClient streaming methods."""

    @pytest.fixture()
    def client(self) -> OllamaClient:
        """Create a client with known parameters."""
        return OllamaClient(base_url="http://localhost:11434", model="test-model")

    @pytest.mark.asyncio
    async def test_generate_stream_yields_tokens(self, client: OllamaClient) -> None:
        """generate_stream should yield token strings from streaming response."""
        # Simulate streaming lines
        lines = [
            '{"response": "Hello", "done": false}',
            '{"response": " world", "done": false}',
            '{"response": "!", "done": true}',
        ]

        # Create a mock async context manager for stream
        mock_stream_response = AsyncMock()
        mock_stream_response.raise_for_status = MagicMock()
        mock_stream_response.aiter_lines = self._async_line_iter(lines)

        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_response)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        with patch.object(client._client, "stream", return_value=mock_stream_ctx):
            tokens = []
            async for token in client.generate_stream(prompt="Hello"):
                tokens.append(token)

        assert tokens == ["Hello", " world", "!"]

    @staticmethod
    def _async_line_iter(lines: list[str]):
        """Create an async iterator function that yields lines."""

        async def _iter():
            for line in lines:
                yield line

        return _iter


class TestOllamaClientRetry:
    """Tests for retry behavior on connection errors."""

    @pytest.fixture()
    def client(self) -> OllamaClient:
        """Create a client with known parameters."""
        return OllamaClient(base_url="http://localhost:11434", model="test-model")

    @pytest.mark.asyncio
    async def test_retry_on_connection_error_then_success(self, client: OllamaClient) -> None:
        """Should retry on ConnectError and succeed on subsequent attempt."""
        success_response = httpx.Response(
            status_code=200,
            json={
                "model": "test-model",
                "response": "recovered",
                "done": True,
                "prompt_eval_count": 5,
                "eval_count": 3,
            },
            request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        )

        call_count = 0

        async def mock_post(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise httpx.ConnectError("Connection refused")
            return success_response

        with patch.object(client._client, "post", side_effect=mock_post):
            result = await client.generate(prompt="test")

        assert call_count == 2
        assert result.text == "recovered"

    @pytest.mark.asyncio
    async def test_retry_exhausted_raises(self, client: OllamaClient) -> None:
        """Should raise after exhausting retry attempts."""
        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")

            with pytest.raises(httpx.ConnectError):
                await client.generate(prompt="test")

        # Should have been called 3 times (initial + 2 retries)
        assert mock_post.call_count == 3
