"""LLM provider factory — unified interface for all inference backends."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from config.settings import settings
from utils.logging import get_logger

if TYPE_CHECKING:
    from inference.cloud_clients import BaseCloudClient
    from inference.ollama_client import OllamaClient

logger = get_logger(__name__)


class LLMResponse(BaseModel):
    """Universal response model returned by all LLM providers.

    Attributes:
        text: Generated text content.
        model: Model identifier used for generation.
        provider: Provider name (ollama, groq, openai, anthropic).
        usage: Token usage counts if available (prompt_tokens, completion_tokens, total_tokens).
        latency_ms: Response time in milliseconds.
        metadata: Any extra provider-specific information.
    """

    text: str
    model: str
    provider: str
    usage: dict = Field(default_factory=dict)
    latency_ms: float = 0.0
    metadata: dict = Field(default_factory=dict)


# Module-level client cache to avoid creating/closing clients per request
_client_cache: dict[str, OllamaClient | BaseCloudClient] = {}


def get_llm(
    provider: str | None = None, model: str | None = None
) -> OllamaClient | BaseCloudClient:
    """Get or create an LLM client for the specified provider.

    Clients are cached and reused across requests to avoid connection
    overhead. The cache key includes both provider and model.

    Args:
        provider: Provider name ("ollama", "groq", "openai", "anthropic").
            Defaults to ``settings.default_provider``.
        model: Model identifier override. Uses provider-specific defaults if None.

    Returns:
        A cached or newly created client instance ready for generation.

    Raises:
        ValueError: If a cloud provider is requested but its API key is not configured.
    """
    from inference.cloud_clients import AnthropicClient, GroqClient, OpenAIClient
    from inference.ollama_client import OllamaClient

    provider = provider or settings.default_provider
    model = model or _get_default_model(provider)

    cache_key = f"{provider}:{model}"
    if cache_key in _client_cache:
        return _client_cache[cache_key]

    client: OllamaClient | BaseCloudClient
    if provider == "ollama":
        client = OllamaClient(model=model)
    elif provider == "groq":
        if not settings.groq_api_key:
            raise ValueError("Groq API key not configured (set SAR_GROQ_API_KEY)")
        client = GroqClient(api_key=settings.groq_api_key, model=model)
    elif provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured (set SAR_OPENAI_API_KEY)")
        client = OpenAIClient(api_key=settings.openai_api_key, model=model)
    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured (set SAR_ANTHROPIC_API_KEY)")
        client = AnthropicClient(api_key=settings.anthropic_api_key, model=model)
    else:
        raise ValueError(f"Unknown LLM provider: {provider!r}")

    _client_cache[cache_key] = client
    logger.info("llm_client_cached", provider=provider, model=model)
    return client


def _get_default_model(provider: str) -> str:
    """Get the default model for a provider."""
    defaults: dict[str, str] = {
        "ollama": settings.llm_model,
        "groq": "llama-3.3-70b-versatile",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-20250514",
    }
    return defaults.get(provider, settings.llm_model)


def clear_llm_cache() -> None:
    """Clear the LLM client cache.

    Call this when configuration changes (e.g., API keys rotated) to
    force recreation of clients on next use.
    """
    global _client_cache
    count = len(_client_cache)
    for client in _client_cache.values():
        if hasattr(client, "close"):
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    _ = loop.create_task(client.close())  # noqa: RUF006
                else:
                    loop.run_until_complete(client.close())
            except Exception:
                pass
    _client_cache.clear()
    logger.info("llm_client_cache_cleared", count=count)


async def generate(
    provider: str | None = None,
    prompt: str = "",
    system_prompt: str = "",
    model: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Convenience function: create a client, generate a response, and close.

    Measures end-to-end latency and stores it in the returned LLMResponse.

    Args:
        provider: Provider name. Defaults to settings.default_provider.
        prompt: The user prompt to send.
        system_prompt: Optional system prompt for context.
        model: Model override.
        **kwargs: Additional arguments passed to the client's generate method.

    Returns:
        LLMResponse with generated text and metadata.
    """
    client = get_llm(provider=provider, model=model)
    try:
        start = time.perf_counter()
        response = await client.generate(prompt=prompt, system_prompt=system_prompt, **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = elapsed_ms
        return response
    finally:
        await client.close()


async def chat(
    provider: str | None = None,
    messages: list[dict] | None = None,
    model: str | None = None,
    **kwargs,
) -> LLMResponse:
    """Convenience function for chat completions.

    Args:
        provider: Provider name. Defaults to settings.default_provider.
        messages: List of message dicts with 'role' and 'content' keys.
        model: Model override.
        **kwargs: Additional arguments passed to the client's chat method.

    Returns:
        LLMResponse with generated text and metadata.
    """
    client = get_llm(provider=provider, model=model)
    try:
        start = time.perf_counter()
        response = await client.chat(messages=messages or [], **kwargs)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.latency_ms = elapsed_ms
        return response
    finally:
        await client.close()
