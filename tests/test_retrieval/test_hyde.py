"""Tests for HyDE (Hypothetical Document Embeddings) module."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from retrieval.hyde import generate_hyde_passage


class TestGenerateHydePassage:
    """Tests for generate_hyde_passage."""

    @patch("retrieval.hyde.call_llm_async", new_callable=AsyncMock)
    async def test_success_returns_concatenated_query_and_passage(self, mock_llm):
        """On success, the hypothetical passage is appended to the original query."""
        mock_llm.return_value = "  This is the hypothetical passage.  "

        result = await generate_hyde_passage("What is RAG?")

        assert result == "What is RAG?\n\nThis is the hypothetical passage."
        mock_llm.assert_awaited_once()
        prompt = mock_llm.call_args[0][0]
        assert "What is RAG?" in prompt

    @patch("retrieval.hyde.call_llm_async", new_callable=AsyncMock)
    async def test_empty_passage_falls_back_to_query(self, mock_llm):
        """If the LLM returns only whitespace, fall back to the raw query."""
        mock_llm.return_value = "   \n\t  "

        result = await generate_hyde_passage("complex query")

        assert result == "complex query"

    @patch("retrieval.hyde.call_llm_async", new_callable=AsyncMock)
    async def test_exception_falls_back_to_query(self, mock_llm):
        """Any LLM failure must not break retrieval — fall back to raw query."""
        mock_llm.side_effect = RuntimeError("ollama timeout")

        result = await generate_hyde_passage("another query")

        assert result == "another query"

    @patch("retrieval.hyde.call_llm_async", new_callable=AsyncMock)
    async def test_routing_params_passed_through(self, mock_llm):
        """Sensitivity and prefer_cloud are forwarded to the inference router."""
        mock_llm.return_value = "passage"

        await generate_hyde_passage(
            "q",
            sensitivity_level="high",
            prefer_cloud=True,
        )

        _, kwargs = mock_llm.call_args
        assert kwargs["sensitivity_level"] == "high"
        assert kwargs["prefer_cloud"] is True

    @patch("retrieval.hyde.call_llm_async", new_callable=AsyncMock)
    async def test_passage_logged_with_length(self, mock_llm):
        """A successful generation is logged with character count."""
        mock_llm.return_value = "short passage"

        with patch("retrieval.hyde.logger") as mock_log:
            await generate_hyde_passage("test")
            mock_log.info.assert_called_once()
            assert mock_log.info.call_args.kwargs["chars"] == len("short passage")
