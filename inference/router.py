"""Sensitivity-based inference routing — keeps sensitive data local."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from config.settings import settings
from inference.llm_factory import LLMResponse, get_llm
from ingestion.metadata import SensitivityLevel
from utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = get_logger(__name__)


class RoutingDecision(BaseModel):
    """Result of the routing logic indicating which provider to use.

    Attributes:
        provider: Selected provider name.
        model: Selected model identifier.
        reason: Human-readable explanation for the routing decision.
        forced_local: Whether local inference was forced due to data sensitivity.
    """

    provider: str
    model: str
    reason: str
    forced_local: bool = False


class InferenceRouter:
    """Routes inference requests based on data sensitivity level.

    Ensures sensitive data never leaves the local environment by routing
    HIGH sensitivity requests exclusively to Ollama (local inference).

    Args:
        default_provider: Default provider when no preference is specified.
            Defaults to settings.default_provider.
        cloud_provider: Preferred cloud provider for low-sensitivity requests.
            Defaults to settings.cloud_provider.
        force_local_for_sensitive: Whether to enforce local-only for HIGH sensitivity.
    """

    def __init__(
        self,
        default_provider: str | None = None,
        cloud_provider: str | None = None,
        force_local_for_sensitive: bool = True,
    ) -> None:
        self.default_provider = default_provider or settings.default_provider
        self.cloud_provider = cloud_provider or settings.cloud_provider
        self.force_local_for_sensitive = force_local_for_sensitive

    def route(
        self,
        sensitivity_level: SensitivityLevel | str,
        prefer_cloud: bool = False,
        override_provider: str | None = None,
    ) -> RoutingDecision:
        """Determine which provider to use based on sensitivity and preferences.

        Routing logic (in priority order):
            1. If override_provider is set, use it (admin override).
            2. If sensitivity is HIGH, ALWAYS use local (Ollama).
            3. If sensitivity is MEDIUM and prefer_cloud is False, use local.
            4. If sensitivity is LOW and prefer_cloud is True and cloud is configured, use cloud.
            5. Default: use local (Ollama).

        Args:
            sensitivity_level: Data sensitivity classification.
            prefer_cloud: Whether the caller prefers cloud inference.
            override_provider: Admin override to force a specific provider.

        Returns:
            RoutingDecision with selected provider and reasoning.
        """
        # Normalize sensitivity level
        if isinstance(sensitivity_level, str):
            sensitivity_level = SensitivityLevel(sensitivity_level.lower())

        # 1. Admin override
        if override_provider:
            model = self._get_model_for_provider(override_provider)
            return RoutingDecision(
                provider=override_provider,
                model=model,
                reason=f"Admin override to provider: {override_provider}",
                forced_local=False,
            )

        # 2. HIGH sensitivity -> always local UNLESS the deploy explicitly opts
        # out via SAR_ALLOW_CLOUD_FOR_HIGH (set in HF Space production image
        # where local Ollama is unavailable; visitor is warned client-side).
        if sensitivity_level == SensitivityLevel.HIGH and self.force_local_for_sensitive:
            if settings.allow_cloud_for_high and self.cloud_provider and self._is_provider_configured(self.cloud_provider):
                model = self._get_model_for_provider(self.cloud_provider)
                return RoutingDecision(
                    provider=self.cloud_provider,
                    model=model,
                    reason=(
                        f"HIGH sensitivity — SAR_ALLOW_CLOUD_FOR_HIGH=true, "
                        f"using {self.cloud_provider}"
                    ),
                    forced_local=False,
                )
            return RoutingDecision(
                provider="ollama",
                model=settings.llm_model,
                reason="HIGH sensitivity data — forced to local inference for privacy",
                forced_local=True,
            )

        # 3. MEDIUM sensitivity -> local by default unless cloud preferred
        if sensitivity_level == SensitivityLevel.MEDIUM:
            if not prefer_cloud:
                return RoutingDecision(
                    provider="ollama",
                    model=settings.llm_model,
                    reason="MEDIUM sensitivity data — using local inference by default",
                    forced_local=False,
                )
            # MEDIUM + prefer_cloud: allow cloud if configured
            if self.cloud_provider and self._is_provider_configured(self.cloud_provider):
                model = self._get_model_for_provider(self.cloud_provider)
                return RoutingDecision(
                    provider=self.cloud_provider,
                    model=model,
                    reason=(
                        f"MEDIUM sensitivity with cloud preference — using {self.cloud_provider}"
                    ),
                    forced_local=False,
                )
            return RoutingDecision(
                provider="ollama",
                model=settings.llm_model,
                reason="MEDIUM sensitivity — cloud preferred but not configured, using local",
                forced_local=False,
            )

        # 4. LOW sensitivity + prefer_cloud + cloud configured
        if (
            sensitivity_level == SensitivityLevel.LOW
            and prefer_cloud
            and self.cloud_provider
            and self._is_provider_configured(self.cloud_provider)
        ):
            model = self._get_model_for_provider(self.cloud_provider)
            return RoutingDecision(
                provider=self.cloud_provider,
                model=model,
                reason=(f"LOW sensitivity with cloud preference — using {self.cloud_provider}"),
                forced_local=False,
            )

        # 5. Default: local
        return RoutingDecision(
            provider="ollama",
            model=settings.llm_model,
            reason="Default routing — using local Ollama inference",
            forced_local=False,
        )

    async def generate_with_routing(
        self,
        prompt: str,
        system_prompt: str = "",
        sensitivity_level: SensitivityLevel | str = "low",
        prefer_cloud: bool = False,
        **kwargs,
    ) -> tuple[LLMResponse, RoutingDecision]:
        """Generate a response with automatic provider routing based on sensitivity.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system context.
            sensitivity_level: Data sensitivity classification.
            prefer_cloud: Whether the caller prefers cloud inference.
            **kwargs: Additional arguments passed to the client's generate method.

        Returns:
            Tuple of (LLMResponse, RoutingDecision).
        """
        decision = self.route(sensitivity_level=sensitivity_level, prefer_cloud=prefer_cloud)
        logger.info(
            "inference_routing",
            provider=decision.provider,
            model=decision.model,
            reason=decision.reason,
            forced_local=decision.forced_local,
        )

        import time

        start = time.perf_counter()
        try:
            client = get_llm(provider=decision.provider, model=decision.model)
            response = await client.generate(prompt=prompt, system_prompt=system_prompt, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.latency_ms = elapsed_ms
            return response, decision
        except Exception as exc:
            # Cloud-fallback when local Ollama is unreachable AND sensitivity
            # allows it (NOT HIGH and NOT forced_local). Tries the configured
            # cloud_provider; if that's also unreachable, re-raises original.
            allow_failover = (
                decision.provider == "ollama"
                and not decision.forced_local
                and self.cloud_provider
                and self._is_provider_configured(self.cloud_provider)
                and self._normalised_sensitivity(sensitivity_level) != SensitivityLevel.HIGH
            )
            if not allow_failover:
                raise
            logger.warning(
                "local_inference_failed_falling_back_to_cloud",
                cloud_provider=self.cloud_provider,
                error=str(exc),
            )
            fallback_model = self._get_model_for_provider(self.cloud_provider)
            fallback_decision = RoutingDecision(
                provider=self.cloud_provider,
                model=fallback_model,
                reason=(f"Local inference failed ({exc!s}); falling back to {self.cloud_provider}"),
                forced_local=False,
            )
            fallback_client = get_llm(
                provider=fallback_decision.provider, model=fallback_decision.model
            )
            start = time.perf_counter()
            response = await fallback_client.generate(
                prompt=prompt, system_prompt=system_prompt, **kwargs
            )
            response.latency_ms = (time.perf_counter() - start) * 1000
            return response, fallback_decision

    @staticmethod
    def _normalised_sensitivity(level: SensitivityLevel | str) -> SensitivityLevel:
        """Coerce a sensitivity input into the enum so comparisons work."""
        if isinstance(level, str):
            try:
                return SensitivityLevel(level.lower())
            except ValueError:
                return SensitivityLevel.LOW
        return level

    async def chat_with_routing(
        self,
        messages: list[dict],
        sensitivity_level: SensitivityLevel | str = "low",
        prefer_cloud: bool = False,
        **kwargs,
    ) -> tuple[LLMResponse, RoutingDecision]:
        """Send a chat request with automatic provider routing based on sensitivity.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            sensitivity_level: Data sensitivity classification.
            prefer_cloud: Whether the caller prefers cloud inference.
            **kwargs: Additional arguments passed to the client's chat method.

        Returns:
            Tuple of (LLMResponse, RoutingDecision).
        """
        decision = self.route(sensitivity_level=sensitivity_level, prefer_cloud=prefer_cloud)
        logger.info(
            "inference_routing",
            provider=decision.provider,
            model=decision.model,
            reason=decision.reason,
            forced_local=decision.forced_local,
        )

        client = get_llm(provider=decision.provider, model=decision.model)
        try:
            import time

            start = time.perf_counter()
            response = await client.chat(messages=messages, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            response.latency_ms = elapsed_ms
            return response, decision
        finally:
            # Clients are cached — do NOT close per-request
            pass

    async def generate_stream_with_routing(
        self,
        prompt: str,
        system_prompt: str = "",
        sensitivity_level: SensitivityLevel | str = "low",
        prefer_cloud: bool = False,
        **kwargs,
    ) -> AsyncGenerator[str, None]:
        """Stream a completion with automatic provider routing.

        All supported providers (Ollama, Groq, OpenAI, Anthropic) implement
        true streaming via their respective SSE/HTTP2 streaming APIs. The
        routing decision determines which provider handles the stream.

        Args:
            prompt: The user prompt text.
            system_prompt: Optional system context.
            sensitivity_level: Data sensitivity classification.
            prefer_cloud: Whether the caller prefers cloud inference.
            **kwargs: Additional arguments passed to the client.

        Yields:
            Token strings as they are generated by the selected provider.
        """
        decision = self.route(sensitivity_level=sensitivity_level, prefer_cloud=prefer_cloud)
        logger.info(
            "inference_stream_routing",
            provider=decision.provider,
            model=decision.model,
            reason=decision.reason,
            forced_local=decision.forced_local,
        )

        client = get_llm(provider=decision.provider, model=decision.model)
        try:
            if hasattr(client, "generate_stream"):
                async for token in client.generate_stream(
                    prompt=prompt, system_prompt=system_prompt, **kwargs
                ):
                    yield token
            else:
                # Fallback: non-streaming, yield full response as single chunk
                response = await client.generate(
                    prompt=prompt, system_prompt=system_prompt, **kwargs
                )
                yield response.text
        finally:
            # Clients are cached — do NOT close per-request
            pass

    def get_available_providers(self) -> list[str]:
        """Return a list of currently configured and available providers.

        A provider is considered available if its required configuration
        (API key for cloud providers) is present.

        Returns:
            List of available provider name strings.
        """
        providers: list[str] = ["ollama"]  # Ollama is always available (local)

        if settings.groq_api_key:
            providers.append("groq")
        if settings.openai_api_key:
            providers.append("openai")
        if settings.anthropic_api_key:
            providers.append("anthropic")

        return providers

    def _is_provider_configured(self, provider: str) -> bool:
        """Check if a provider has its required configuration set.

        Args:
            provider: Provider name to check.

        Returns:
            True if the provider is properly configured.
        """
        if provider == "ollama":
            return True
        if provider == "groq":
            return bool(settings.groq_api_key)
        if provider == "openai":
            return bool(settings.openai_api_key)
        if provider == "anthropic":
            return bool(settings.anthropic_api_key)
        return False

    def _get_model_for_provider(self, provider: str) -> str:
        """Get the default model identifier for a given provider.

        Args:
            provider: Provider name.

        Returns:
            Default model string for the provider.
        """
        model_defaults: dict[str, str] = {
            "ollama": settings.llm_model,
            "groq": "llama-3.3-70b-versatile",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-sonnet-4-20250514",
        }
        return model_defaults.get(provider, settings.llm_model)
