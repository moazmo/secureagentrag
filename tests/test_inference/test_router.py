"""Tests for the inference router — sensitivity-based provider routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from inference.llm_factory import LLMResponse
from inference.router import InferenceRouter, RoutingDecision


class TestRoutingDecision:
    """Tests for the RoutingDecision model."""

    def test_create_decision(self) -> None:
        """RoutingDecision should hold routing information."""
        decision = RoutingDecision(
            provider="ollama",
            model="qwen3:8b",
            reason="Default routing",
            forced_local=False,
        )
        assert decision.provider == "ollama"
        assert decision.model == "qwen3:8b"
        assert decision.forced_local is False

    def test_forced_local_default(self) -> None:
        """forced_local should default to False."""
        decision = RoutingDecision(provider="ollama", model="test", reason="test")
        assert decision.forced_local is False


class TestInferenceRouterRouting:
    """Tests for InferenceRouter.route() logic."""

    @pytest.fixture()
    def router_with_cloud(self) -> InferenceRouter:
        """Create a router with cloud provider configured."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.cloud_provider = "groq"
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "test-key"
            mock_settings.allow_cloud_for_high = False
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            router = InferenceRouter()
        return router

    @pytest.fixture()
    def router_no_cloud(self) -> InferenceRouter:
        """Create a router with no cloud provider configured."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.cloud_provider = None
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = None
            mock_settings.allow_cloud_for_high = False
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            router = InferenceRouter()
        return router

    def test_high_sensitivity_always_local(self, router_with_cloud: InferenceRouter) -> None:
        """HIGH sensitivity should ALWAYS route to local, forced_local=True."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "test-key"
            mock_settings.allow_cloud_for_high = False
            decision = router_with_cloud.route(sensitivity_level="high", prefer_cloud=True)

        assert decision.provider == "ollama"
        assert decision.forced_local is True
        assert "HIGH" in decision.reason

    def test_high_sensitivity_ignores_prefer_cloud(
        self, router_with_cloud: InferenceRouter
    ) -> None:
        """HIGH sensitivity should ignore prefer_cloud flag."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "test-key"
            mock_settings.allow_cloud_for_high = False
            decision = router_with_cloud.route(sensitivity_level="high", prefer_cloud=True)

        assert decision.provider == "ollama"
        assert decision.forced_local is True

    def test_high_sensitivity_cloud_unlock(
        self, router_with_cloud: InferenceRouter
    ) -> None:
        """SAR_ALLOW_CLOUD_FOR_HIGH=True routes HIGH to the cloud provider.

        Production HF Space deploys have no local Ollama; the opt-in flag
        permits cloud synthesis on HIGH-classified chunks. The frontend
        labels the answer "sensitive: routed to cloud" so the visitor is
        aware. forced_local stays False because the request did leave
        the local environment.
        """
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "test-key"
            mock_settings.allow_cloud_for_high = True
            decision = router_with_cloud.route(sensitivity_level="high", prefer_cloud=False)

        assert decision.provider == "groq"
        assert decision.forced_local is False
        assert "SAR_ALLOW_CLOUD_FOR_HIGH" in decision.reason

    def test_low_sensitivity_prefer_cloud(self, router_with_cloud: InferenceRouter) -> None:
        """LOW sensitivity + prefer_cloud + cloud configured -> use cloud."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "test-key"
            mock_settings.allow_cloud_for_high = False
            decision = router_with_cloud.route(sensitivity_level="low", prefer_cloud=True)

        assert decision.provider == "groq"
        assert decision.forced_local is False
        assert "LOW" in decision.reason

    def test_low_sensitivity_no_cloud_preference(self, router_with_cloud: InferenceRouter) -> None:
        """LOW sensitivity without prefer_cloud -> use default (local)."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            decision = router_with_cloud.route(sensitivity_level="low", prefer_cloud=False)

        assert decision.provider == "ollama"
        assert decision.forced_local is False

    def test_medium_sensitivity_default_local(self, router_with_cloud: InferenceRouter) -> None:
        """MEDIUM sensitivity without prefer_cloud -> use local."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            decision = router_with_cloud.route(sensitivity_level="medium", prefer_cloud=False)

        assert decision.provider == "ollama"
        assert decision.forced_local is False
        assert "MEDIUM" in decision.reason

    def test_medium_sensitivity_prefer_cloud_with_config(
        self, router_with_cloud: InferenceRouter
    ) -> None:
        """MEDIUM sensitivity + prefer_cloud + cloud configured -> use cloud."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "test-key"
            mock_settings.allow_cloud_for_high = False
            decision = router_with_cloud.route(sensitivity_level="medium", prefer_cloud=True)

        assert decision.provider == "groq"
        assert decision.forced_local is False

    def test_override_provider_bypasses_all_logic(self, router_with_cloud: InferenceRouter) -> None:
        """override_provider should bypass sensitivity routing entirely."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            decision = router_with_cloud.route(
                sensitivity_level="high",
                prefer_cloud=False,
                override_provider="openai",
            )

        assert decision.provider == "openai"
        assert decision.forced_local is False
        assert "Admin override" in decision.reason

    def test_low_sensitivity_prefer_cloud_no_config(self, router_no_cloud: InferenceRouter) -> None:
        """LOW + prefer_cloud but no cloud configured -> fallback to local."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = None
            mock_settings.allow_cloud_for_high = False
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            decision = router_no_cloud.route(sensitivity_level="low", prefer_cloud=True)

        assert decision.provider == "ollama"


class TestInferenceRouterAvailableProviders:
    """Tests for InferenceRouter.get_available_providers()."""

    def test_only_ollama_when_no_keys(self) -> None:
        """Should return only ollama when no API keys are configured."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.cloud_provider = None
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = None
            mock_settings.allow_cloud_for_high = False
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            router = InferenceRouter()
            providers = router.get_available_providers()

        assert providers == ["ollama"]

    def test_all_providers_with_keys(self) -> None:
        """Should return all providers when all API keys are configured."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.cloud_provider = "groq"
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "key1"
            mock_settings.allow_cloud_for_high = False
            mock_settings.openai_api_key = "key2"
            mock_settings.anthropic_api_key = "key3"
            router = InferenceRouter()
            providers = router.get_available_providers()

        assert "ollama" in providers
        assert "groq" in providers
        assert "openai" in providers
        assert "anthropic" in providers

    def test_partial_keys(self) -> None:
        """Should return only configured providers."""
        with patch("inference.router.settings") as mock_settings:
            mock_settings.default_provider = "ollama"
            mock_settings.cloud_provider = "groq"
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "key1"
            mock_settings.allow_cloud_for_high = False
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = "key3"
            router = InferenceRouter()
            providers = router.get_available_providers()

        assert "ollama" in providers
        assert "groq" in providers
        assert "openai" not in providers
        assert "anthropic" in providers


class TestInferenceRouterGenerateWithRouting:
    """Tests for InferenceRouter.generate_with_routing()."""

    @pytest.mark.asyncio
    async def test_generate_with_routing_calls_correct_provider(self) -> None:
        """generate_with_routing should route and call the selected provider."""
        mock_response = LLMResponse(text="Local response", model="qwen3:8b", provider="ollama")
        mock_client = AsyncMock()
        mock_client.generate = AsyncMock(return_value=mock_response)
        mock_client.close = AsyncMock()

        with (
            patch("inference.router.settings") as mock_settings,
            patch("inference.router.get_llm", return_value=mock_client),
        ):
            mock_settings.default_provider = "ollama"
            mock_settings.cloud_provider = "groq"
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "key"
            mock_settings.allow_cloud_for_high = False

            router = InferenceRouter()
            response, decision = await router.generate_with_routing(
                prompt="Hello",
                sensitivity_level="high",
            )

        assert decision.provider == "ollama"
        assert decision.forced_local is True
        assert response.text == "Local response"
        # With client caching, close is no longer called per-request

    @pytest.mark.asyncio
    async def test_chat_with_routing_calls_correct_provider(self) -> None:
        """chat_with_routing should route and call the selected provider."""
        mock_response = LLMResponse(
            text="Cloud response", model="llama-3.3-70b-versatile", provider="groq"
        )
        mock_client = AsyncMock()
        mock_client.chat = AsyncMock(return_value=mock_response)
        mock_client.close = AsyncMock()

        with (
            patch("inference.router.settings") as mock_settings,
            patch("inference.router.get_llm", return_value=mock_client),
        ):
            mock_settings.default_provider = "ollama"
            mock_settings.cloud_provider = "groq"
            mock_settings.llm_model = "qwen3:8b"
            mock_settings.groq_api_key = "key"
            mock_settings.allow_cloud_for_high = False

            router = InferenceRouter()
            messages = [{"role": "user", "content": "Hello"}]
            response, decision = await router.chat_with_routing(
                messages=messages,
                sensitivity_level="low",
                prefer_cloud=True,
            )

        assert decision.provider == "groq"
        assert response.text == "Cloud response"
        # With client caching, close is no longer called per-request
