"""Tests for self-query retrieval module."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

from retrieval.self_query import (
    build_qdrant_filter_conditions,
    extract_self_query_filters,
)


class TestExtractSelfQueryFilters:
    """Tests for extract_self_query_filters."""

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_extracts_source_file(self, mock_llm):
        """A query mentioning a filename yields a source_file filter."""
        mock_llm.return_value = json.dumps({"source_file": "report.pdf"})

        result = await extract_self_query_filters("What does report.pdf say?")

        assert result == {"source_file": "report.pdf"}
        mock_llm.assert_awaited_once()
        assert "What does report.pdf say?" in mock_llm.call_args[0][0]

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_extracts_sensitivity_and_roles(self, mock_llm):
        """Multiple filters can be extracted in one call."""
        mock_llm.return_value = json.dumps(
            {"sensitivity_level": "high", "roles": ["admin", "engineer"]}
        )

        result = await extract_self_query_filters("Show high-sensitivity admin docs")

        assert result["sensitivity_level"] == "high"
        assert result["roles"] == ["admin", "engineer"]

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_date_filters_parsed(self, mock_llm):
        """ISO date strings are accepted for date_after / date_before."""
        mock_llm.return_value = json.dumps(
            {"date_after": "2024-01-01", "date_before": "2024-12-31"}
        )

        result = await extract_self_query_filters("Docs from 2024")

        assert result["date_after"] == "2024-01-01"
        assert result["date_before"] == "2024-12-31"

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_invalid_date_skipped(self, mock_llm):
        """Malformed dates are dropped rather than crashing."""
        mock_llm.return_value = json.dumps({"date_after": "not-a-date"})

        result = await extract_self_query_filters("Docs from invalid date")

        assert "date_after" not in result

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_empty_dict_when_no_filters(self, mock_llm):
        """If the LLM returns {}, we get an empty dict."""
        mock_llm.return_value = "{}"

        result = await extract_self_query_filters("What is RAG?")

        assert result == {}

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_strips_markdown_fences(self, mock_llm):
        """Some models wrap JSON in ```json ... ``` — we strip it."""
        mock_llm.return_value = '```json\n{"source_file": "x.pdf"}\n```'

        result = await extract_self_query_filters("What about x.pdf?")

        assert result == {"source_file": "x.pdf"}

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_invalid_json_returns_empty(self, mock_llm):
        """Non-JSON response yields empty dict."""
        mock_llm.return_value = "I think the answer is 42"

        result = await extract_self_query_filters("Random query")

        assert result == {}

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_exception_returns_empty(self, mock_llm):
        """LLM failure yields empty dict so retrieval continues."""
        mock_llm.side_effect = RuntimeError("timeout")

        result = await extract_self_query_filters("fail me")

        assert result == {}

    @patch("retrieval.self_query.call_llm_async", new_callable=AsyncMock)
    async def test_none_values_skipped(self, mock_llm):
        """Null or empty values are not included in the result."""
        mock_llm.return_value = json.dumps(
            {"source_file": "a.pdf", "org_id": None, "sensitivity_level": ""}
        )

        result = await extract_self_query_filters("mixed query")

        assert result == {"source_file": "a.pdf"}


class TestBuildQdrantFilterConditions:
    """Tests for build_qdrant_filter_conditions."""

    def test_source_file_condition(self):
        """source_file maps to a MatchValue condition."""
        conditions = build_qdrant_filter_conditions({"source_file": "report.pdf"})
        assert len(conditions) == 1
        assert conditions[0]["key"] == "source_file"

    def test_org_id_condition(self):
        """org_id maps to a MatchValue condition."""
        conditions = build_qdrant_filter_conditions({"org_id": "acme"})
        assert conditions[0]["key"] == "org_id"

    def test_sensitivity_level_mapped_to_int(self):
        """sensitivity_level 'medium' maps to integer 2."""
        conditions = build_qdrant_filter_conditions({"sensitivity_level": "medium"})
        assert conditions[0]["key"] == "sensitivity_level_int"

    def test_unknown_sensitivity_skipped(self):
        """An unrecognised sensitivity label is silently ignored."""
        conditions = build_qdrant_filter_conditions({"sensitivity_level": "extreme"})
        assert conditions == []

    def test_roles_condition(self):
        """roles list maps to MatchAny condition."""
        conditions = build_qdrant_filter_conditions({"roles": ["admin", "viewer"]})
        assert conditions[0]["key"] == "roles"

    def test_date_range_conditions(self):
        """date_after and date_before both produce Range conditions."""
        conditions = build_qdrant_filter_conditions(
            {"date_after": "2024-01-01", "date_before": "2024-06-01"}
        )
        keys = [c["key"] for c in conditions]
        assert keys == ["ingested_at", "ingested_at"]
        # One should have gte, the other lte
        ranges = [c.get("range") for c in conditions]
        assert any(r.gte is not None for r in ranges)
        assert any(r.lte is not None for r in ranges)

    def test_multiple_conditions(self):
        """A filter dict with multiple keys produces multiple conditions."""
        conditions = build_qdrant_filter_conditions(
            {"source_file": "x.pdf", "org_id": "acme", "sensitivity_level": "low"}
        )
        keys = {c["key"] for c in conditions}
        assert keys == {"source_file", "org_id", "sensitivity_level_int"}

    def test_empty_dict(self):
        """An empty filter dict yields an empty condition list."""
        conditions = build_qdrant_filter_conditions({})
        assert conditions == []
