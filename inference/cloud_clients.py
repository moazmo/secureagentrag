"""Cloud LLM provider clients (Groq, OpenAI, Anthropic Claude)."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

import httpx
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


# Retry on transient connection failures AND 429 rate-limit responses.
# Groq's free tier is 30 RPM; a single user query can fire grader +
# synth + faith calls that exceed that bucket. We honour the Retry-After
# header where present, fall back to exponential backoff otherwise.
class _RateLimitError(Exception):
    """Lift a 429 into something tenacity can catch and back off on."""


def _raise_for_status_with_429(resp: httpx.Response) -> None:
    """Like httpx.Response.raise_for_status but lifts 429 to _RateLimitError."""
    if resp.status_code == 429:
        raise _RateLimitError(resp.headers.get("Retry-After", ""))
    resp.raise_for_status()


_retry_on_connection = retry(
    retry=retry_if_exception_type((httpx.ConnectError, httpx.TimeoutException, _RateLimitError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    reraise=True,
)


class LLMProvider(StrEnum):
    """Supported LLM provider identifiers."""

    OLLAMA = "ollama"
    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class BaseCloudClient(ABC):
    """Abstract base class for cloud LLM provider clients.

    Args:
        api_key: Provider API key for authentication.
        model: Default model identifier.
        timeout: Request timeout in seconds.
    """

    def __init__(self, api_key: str, model: str, timeout: float = 60.0) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a completion from the provider.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: When True, request JSON-formatted output.

        Returns:
            LLMResponse with generated text and metadata.
        """

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a chat conversation to the provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated text and metadata.
        """

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion from the provider, yielding tokens as they arrive.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Token strings as they are generated.
        """

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider API is reachable.

        Returns:
            True if the API responds successfully.
        """

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._client.aclose()

    async def __aenter__(self) -> BaseCloudClient:
        """Enter async context manager."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager, closing the client."""
        await self.close()


def make_byok_cloud_client(
    *,
    provider: str,
    user_key: str,
    model: str | None = None,
    timeout: float = 60.0,
) -> BaseCloudClient:
    """Build a per-request cloud LLM client that uses the visitor's API key.

    Each call returns a **fresh client instance** holding the supplied key
    in its own ``self.api_key`` slot. The visitor's key never lands on any
    module-level singleton, never mixes into the owner-key client, and is
    discarded when the FastAPI request scope ends.

    Args:
        provider: One of ``"groq"`` / ``"openai"`` / ``"anthropic"``.
        user_key: The visitor-supplied API key from ``X-User-LLM-Key``.
        model: Override the provider's default model.
        timeout: Per-request HTTP timeout in seconds.

    Returns:
        A new ``BaseCloudClient`` subclass instance bound to the visitor key.

    Raises:
        ValueError: ``provider`` is not in the BYOK allowlist or ``user_key``
            is missing.
    """
    if not user_key or not user_key.strip():
        raise ValueError("make_byok_cloud_client called without a user key")
    prov = (provider or "").lower()
    if prov == "groq":
        return GroqClient(
            api_key=user_key.strip(), model=model or "llama-3.1-8b-instant", timeout=timeout
        )
    if prov == "openai":
        return OpenAIClient(api_key=user_key.strip(), model=model or "gpt-4o-mini", timeout=timeout)
    if prov == "anthropic":
        return AnthropicClient(
            api_key=user_key.strip(),
            model=model or "claude-sonnet-4-20250514",
            timeout=timeout,
        )
    raise ValueError(f"BYOK provider not supported: {provider!r}")


class OpenAICompatibleClient(BaseCloudClient):
    """Shared client for OpenAI Chat Completions-compatible APIs.

    Both Groq and OpenAI implement the same wire format
    (``POST /chat/completions`` + SSE streaming). Subclasses supply only
    the ``api_base`` URL and the ``provider`` tag — every method on
    ``BaseCloudClient`` is implemented once, here, and inherited.
    """

    #: Subclasses override these two class attrs.
    api_base: str = ""
    provider_name: str = ""

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _messages(prompt: str, system_prompt: str) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        if system_prompt:
            out.append({"role": "system", "content": system_prompt})
        out.append({"role": "user", "content": prompt})
        return out

    @_retry_on_connection
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        return await self.chat(
            messages=self._messages(prompt, system_prompt),
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
        )

    @_retry_on_connection
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        response = await self._client.post(
            f"{self.api_base}/chat/completions",
            headers=self._headers(),
            json=payload,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        _raise_for_status_with_429(response)

        data = response.json()
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = data.get("usage", {})

        return LLMResponse(
            text=message.get("content", ""),
            model=data.get("model", self.model),
            provider=self.provider_name,
            usage={
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            },
            latency_ms=elapsed_ms,
        )

    @_retry_on_connection
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._messages(prompt, system_prompt),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        async with self._client.stream(
            "POST",
            f"{self.api_base}/chat/completions",
            headers={**self._headers(), "Accept": "text/event-stream"},
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choice = data.get("choices", [{}])[0]
                token = choice.get("delta", {}).get("content", "")
                if token:
                    yield token

    @_retry_on_connection
    async def health_check(self) -> bool:
        try:
            response = await self._client.get(f"{self.api_base}/models", headers=self._headers())
            return response.status_code in (200, 401)
        except (httpx.ConnectError, httpx.TimeoutException):
            return False


class GroqClient(OpenAICompatibleClient):
    """Groq cloud LLM client (OpenAI-compatible API at api.groq.com)."""

    provider_name = "groq"

    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        timeout: float = 60.0,
    ) -> None:
        super().__init__(api_key=api_key, model=model, timeout=timeout)
        self.api_base = settings.groq_api_base


class OpenAIClient(OpenAICompatibleClient):
    """OpenAI cloud LLM client (Chat Completions API at api.openai.com)."""

    provider_name = "openai"

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        timeout: float = 60.0,
    ) -> None:
        super().__init__(api_key=api_key, model=model, timeout=timeout)
        self.api_base = settings.openai_api_base


class AnthropicClient(BaseCloudClient):
    """Anthropic Claude cloud LLM client using the Messages API.

    Args:
        api_key: Anthropic API key.
        model: Model identifier. Defaults to "claude-sonnet-4-20250514".
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        timeout: float = 60.0,
    ) -> None:
        super().__init__(api_key=api_key, model=model, timeout=timeout)
        self._api_base = settings.anthropic_api_base

    def _headers(self) -> dict[str, str]:
        """Build request headers with Anthropic-specific authentication."""
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    @_retry_on_connection
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Generate a completion via Anthropic's Messages API.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            json_mode: Anthropic does not support native JSON mode; ignored.

        Returns:
            LLMResponse with generated text and metadata.
        """
        messages: list[dict[str, str]] = [{"role": "user", "content": prompt}]
        return await self._send_messages(
            messages=messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @_retry_on_connection
    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Send a chat request to Anthropic's Messages API.

        Anthropic uses a separate 'system' parameter instead of a system message
        in the messages list. This method extracts any system message and handles
        the format conversion.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated text and metadata.
        """
        # Extract system message if present
        system_prompt = ""
        anthropic_messages: list[dict[str, str]] = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                anthropic_messages.append(msg)

        return await self._send_messages(
            messages=anthropic_messages,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _send_messages(
        self,
        messages: list[dict],
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Internal method to send messages to Anthropic's API.

        Args:
            messages: Anthropic-formatted messages (no system role).
            system_prompt: System prompt passed as top-level parameter.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            LLMResponse with generated text and metadata.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system_prompt:
            payload["system"] = system_prompt

        start = time.perf_counter()
        response = await self._client.post(
            f"{self._api_base}/messages",
            headers=self._headers(),
            json=payload,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.raise_for_status()

        data = response.json()
        # Anthropic returns content as a list of content blocks
        content_blocks = data.get("content", [])
        text = ""
        for block in content_blocks:
            if block.get("type") == "text":
                text += block.get("text", "")

        usage = data.get("usage", {})
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            provider="anthropic",
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": (usage.get("input_tokens", 0) + usage.get("output_tokens", 0)),
            },
            latency_ms=elapsed_ms,
        )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion via Anthropic's Messages API.

        Anthropic supports streaming via SSE. Yields text content blocks
        as they arrive.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system context.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Yields:
            Token strings as they are generated.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with self._client.stream(
            "POST",
            f"{self._api_base}/messages",
            headers={**self._headers(), "Accept": "text/event-stream"},
            json=payload,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                line = line.strip()
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        event_type = data.get("type", "")
                        if event_type == "content_block_delta":
                            delta = data.get("delta", {})
                            token = delta.get("text", "")
                            if token:
                                yield token
                        elif event_type == "message_stop":
                            break
                    except json.JSONDecodeError:
                        continue

    @_retry_on_connection
    async def health_check(self) -> bool:
        """Check if the Anthropic API is reachable.

        Returns:
            True if the API responds.
        """
        try:
            # Anthropic doesn't have a simple health endpoint; try a minimal request
            response = await self._client.post(
                f"{self._api_base}/messages",
                headers=self._headers(),
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "hi"}],
                    "max_tokens": 1,
                },
            )
            # Any response (even 401) means the service is reachable
            return response.status_code in (200, 401, 400)
        except (httpx.ConnectError, httpx.TimeoutException):
            return False
