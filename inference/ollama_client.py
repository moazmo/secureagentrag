"""Async Ollama client wrapper with streaming support and health checks."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import httpx

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.settings import settings
from inference.llm_factory import LLMResponse
from utils.logging import get_logger

logger = get_logger(__name__)

# Retry decorator for transient connection failures only
_retry_on_connection = retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)


class OllamaClient:
    """Async client for the Ollama local LLM inference server.

    Supports generate (completion), chat, streaming, health checks,
    and model listing via the Ollama HTTP API.

    Args:
        base_url: Ollama server base URL. Defaults to settings.ollama_url.
        model: Default model name. Defaults to settings.llm_model.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url if base_url is not None else settings.ollama_url).rstrip("/")
        self.model = model if model is not None else settings.llm_model
        self.timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

    @_retry_on_connection
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a completion from the Ollama API.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system context.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.
            json_mode: When True, request JSON-formatted output.

        Returns:
            LLMResponse with generated text and metadata.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "keep_alive": settings.ollama_keep_alive,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if json_mode:
            payload["format"] = "json"

        start = time.perf_counter()
        response = await self._client.post("/api/generate", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()

        data = response.json()
        return LLMResponse(
            text=data.get("response", ""),
            model=data.get("model", self.model),
            provider="ollama",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
            },
            latency_ms=elapsed_ms,
            metadata={
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
            },
        )

    @_retry_on_connection
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a chat conversation to the Ollama API.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                Roles: "system", "user", "assistant".
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated text and metadata.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            "keep_alive": settings.ollama_keep_alive,
        }

        start = time.perf_counter()
        response = await self._client.post("/api/chat", json=payload)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()

        data = response.json()
        message = data.get("message", {})
        return LLMResponse(
            text=message.get("content", ""),
            model=data.get("model", self.model),
            provider="ollama",
            usage={
                "prompt_tokens": data.get("prompt_eval_count", 0),
                "completion_tokens": data.get("eval_count", 0),
                "total_tokens": (data.get("prompt_eval_count", 0) + data.get("eval_count", 0)),
            },
            latency_ms=elapsed_ms,
            metadata={
                "total_duration": data.get("total_duration"),
                "load_duration": data.get("load_duration"),
            },
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from the Ollama API, yielding tokens as they arrive.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system context.
            temperature: Sampling temperature (0.0-1.0).

        Yields:
            Token strings as they are generated.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
            "keep_alive": settings.ollama_keep_alive,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with self._client.stream("POST", "/api/generate", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    token = data.get("response", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break

    async def chat_stream(
        self,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion from the Ollama API, yielding tokens as they arrive.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0-1.0).

        Yields:
            Token strings as they are generated.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            },
            "keep_alive": settings.ollama_keep_alive,
        }

        async with self._client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line:
                    import json

                    data = json.loads(line)
                    message = data.get("message", {})
                    token = message.get("content", "")
                    if token:
                        yield token
                    if data.get("done", False):
                        break

    @_retry_on_connection
    async def health_check(self) -> bool:
        """Check if the Ollama server is reachable and responding.

        Returns:
            True if the server responds with HTTP 200, False otherwise.
        """
        try:
            response = await self._client.get("/api/tags")
            return response.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    @_retry_on_connection
    async def list_models(self) -> list[str]:
        """List all models available on the Ollama server.

        Returns:
            List of model name strings.
        """
        response = await self._client.get("/api/tags")
        response.raise_for_status()
        data = response.json()
        models = data.get("models", [])
        return [m.get("name", "") for m in models]

    @_retry_on_connection
    async def get_model_info(self, model: str | None = None) -> dict | None:
        """Get detailed information about a specific model.

        Args:
            model: Model name to query. Defaults to the client's configured model.

        Returns:
            Dict with model info, or None if model not found.
        """
        target_model = model or self.model
        try:
            response = await self._client.post("/api/show", json={"name": target_model})
            if response.status_code == 200:
                return response.json()
            return None
        except httpx.HTTPStatusError:
            return None

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> OllamaClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, closing the client."""
        await self.close()
