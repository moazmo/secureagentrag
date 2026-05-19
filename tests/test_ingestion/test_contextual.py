"""Tests for Anthropic-style contextual retrieval module."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

from ingestion.contextual import (
    _generate_one,
    generate_chunk_contexts,
    merge_chunks,
)


class TestGenerateOne:
    """Tests for _generate_one (single chunk context generation)."""

    @patch("ingestion.contextual.call_llm_async", new_callable=AsyncMock)
    async def test_success_returns_stripped_context(self, mock_llm):
        """A successful LLM call returns the stripped context string."""
        mock_llm.return_value = "  Context about the GOVERN function.  "
        sem = asyncio.Semaphore(10)

        result = await _generate_one(
            document_text="Full doc about NIST AI RMF.",
            chunk_text="GOVERN function details.",
            semaphore=sem,
            prefer_cloud=False,
            max_doc_chars=1000,
        )

        assert result == "Context about the GOVERN function."
        mock_llm.assert_awaited_once()
        prompt = mock_llm.call_args[0][0]
        assert "Full doc about NIST AI RMF." in prompt
        assert "GOVERN function details." in prompt

    @patch("ingestion.contextual.call_llm_async", new_callable=AsyncMock)
    async def test_failure_returns_empty_string(self, mock_llm):
        """An LLM failure returns an empty string so the chunk is used as-is."""
        mock_llm.side_effect = RuntimeError("model unavailable")
        sem = asyncio.Semaphore(10)

        result = await _generate_one(
            document_text="doc",
            chunk_text="chunk",
            semaphore=sem,
            prefer_cloud=False,
            max_doc_chars=100,
        )

        assert result == ""

    @patch("ingestion.contextual.call_llm_async", new_callable=AsyncMock)
    async def test_document_truncated_to_max_doc_chars(self, mock_llm):
        """Long documents are truncated before being placed in the prompt."""
        mock_llm.return_value = "ctx"
        sem = asyncio.Semaphore(10)
        long_doc = "x" * 100_000

        await _generate_one(long_doc, "chunk", sem, False, max_doc_chars=500)

        prompt = mock_llm.call_args[0][0]
        assert len(prompt) < len(long_doc) + 100  # significantly shorter

    @patch("ingestion.contextual.call_llm_async", new_callable=AsyncMock)
    async def test_semaphore_limits_concurrency(self, mock_llm):
        """The semaphore bounds simultaneous LLM calls."""
        mock_llm.return_value = "ctx"
        sem = asyncio.Semaphore(1)  # only 1 concurrent

        # Launch two tasks; the second should wait
        t1 = asyncio.create_task(_generate_one("doc", "c1", sem, False, 100))
        t2 = asyncio.create_task(_generate_one("doc", "c2", sem, False, 100))
        await asyncio.gather(t1, t2)

        assert mock_llm.await_count == 2


class TestGenerateChunkContexts:
    """Tests for generate_chunk_contexts (batch generation)."""

    @patch("ingestion.contextual.call_llm_async", new_callable=AsyncMock)
    async def test_empty_chunks_returns_empty(self, mock_llm):
        """Passing an empty chunk list short-circuits to an empty list."""
        result = await generate_chunk_contexts("doc", [])

        assert result == []
        mock_llm.assert_not_awaited()

    @patch("ingestion.contextual.call_llm_async", new_callable=AsyncMock)
    async def test_multiple_chunks_parallel(self, mock_llm):
        """All chunks receive a context and results preserve order."""
        mock_llm.side_effect = ["ctx1", "ctx2", "ctx3"]

        result = await generate_chunk_contexts(
            "document text",
            ["chunk A", "chunk B", "chunk C"],
        )

        assert result == ["ctx1", "ctx2", "ctx3"]
        assert mock_llm.await_count == 3

    @patch("ingestion.contextual.call_llm_async", new_callable=AsyncMock)
    async def test_partial_failure_preserved(self, mock_llm):
        """If one chunk fails, its slot is empty string; others succeed."""
        mock_llm.side_effect = ["ctx1", RuntimeError("fail"), "ctx3"]

        result = await generate_chunk_contexts("doc", ["c1", "c2", "c3"])

        assert result == ["ctx1", "", "ctx3"]

    @patch("ingestion.contextual.call_llm_async", new_callable=AsyncMock)
    async def test_logged_summary(self, mock_llm):
        """Generation summary is logged with counts."""
        mock_llm.side_effect = ["ctx1", ""]

        with patch("ingestion.contextual.logger") as mock_log:
            await generate_chunk_contexts("doc", ["c1", "c2"])
            mock_log.info.assert_called_once()
            kwargs = mock_log.info.call_args.kwargs
            assert kwargs["chunks"] == 2
            assert kwargs["successful"] == 1


class TestMergeChunks:
    """Tests for merge_chunks."""

    def test_merge_with_context_prefixes_chunk(self):
        """When a context is present, it is prepended to the chunk."""
        chunks = ["chunk1", "chunk2"]
        contexts = ["ctx1", "ctx2"]

        result = merge_chunks(chunks, contexts)

        assert result == ["Context: ctx1\n\nchunk1", "Context: ctx2\n\nchunk2"]

    def test_empty_context_passes_through(self):
        """An empty context leaves the original chunk unchanged."""
        chunks = ["chunk1", "chunk2"]
        contexts = ["", "ctx2"]

        result = merge_chunks(chunks, contexts)

        assert result == ["chunk1", "Context: ctx2\n\nchunk2"]

    def test_mismatched_lengths_raises(self):
        """Mismatched chunk/context counts should raise (zip strict behaviour)."""
        # zip(strict=False) silently drops extras; our test documents this.
        chunks = ["c1", "c2", "c3"]
        contexts = ["ctx1", "ctx2"]

        result = merge_chunks(chunks, contexts)
        assert len(result) == 2  # strict=False truncates to shorter list

    def test_all_empty_contexts(self):
        """If every context is empty, output equals input."""
        chunks = ["a", "b", "c"]
        contexts = ["", "", ""]

        result = merge_chunks(chunks, contexts)

        assert result == chunks
