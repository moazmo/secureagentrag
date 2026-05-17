"""Tests for the answer synthesizer agent."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.agents.synthesizer import (
    _add_disclaimers,
    _extract_citations,
    synthesize_answer,
)
from utils.async_helpers import run_async


@pytest.fixture()
def synth_state():
    """Create a base state for synthesizer tests."""
    return {
        "query": "What is RAG?",
        "user_context": {
            "user_id": "user1",
            "org_id": "org1",
            "roles": ["viewer"],
            "clearance_level": 2,
        },
        "query_type": "simple",
        "rewritten_query": "What is Retrieval-Augmented Generation?",
        "security_passed": True,
        "security_message": "",
        "documents": [
            {
                "doc_id": "d1",
                "text": "RAG combines retrieval with generation for better answers.",
                "score": 0.9,
                "relevant": True,
                "metadata": {
                    "source_file": "rag_intro.pdf",
                    "page_number": 1,
                    "sensitivity_level": "low",
                },
            },
            {
                "doc_id": "d2",
                "text": "Language models can hallucinate without grounding context.",
                "score": 0.8,
                "relevant": True,
                "metadata": {
                    "source_file": "llm_basics.pdf",
                    "page_number": 5,
                    "sensitivity_level": "low",
                },
            },
        ],
        "relevant_documents": [
            {
                "doc_id": "d1",
                "text": "RAG combines retrieval with generation for better answers.",
                "score": 0.9,
                "relevant": True,
                "metadata": {
                    "source_file": "rag_intro.pdf",
                    "page_number": 1,
                    "sensitivity_level": "low",
                },
            },
            {
                "doc_id": "d2",
                "text": "Language models can hallucinate without grounding context.",
                "score": 0.8,
                "relevant": True,
                "metadata": {
                    "source_file": "llm_basics.pdf",
                    "page_number": 5,
                    "sensitivity_level": "low",
                },
            },
        ],
        "relevance_ratio": 1.0,
        "retry_count": 0,
        "max_retries": 2,
        "generation": "",
        "citations": [],
        "confidence_score": 0.0,
        "needs_human_review": False,
        "evaluation_notes": "",
        "audit_trail": [],
    }


class TestSynthesizeAnswer:
    """Tests for the synthesize_answer function."""

    @patch("core.agents.synthesizer.call_llm_async")
    def test_synthesize_answer_with_citations(self, mock_llm, synth_state):
        """Test answer synthesis with proper citations."""
        mock_llm.return_value = (
            "RAG (Retrieval-Augmented Generation) is a technique that combines "
            "information retrieval with language model generation [1]. This helps "
            "reduce hallucinations by grounding responses in retrieved context [2]."
        )

        result = run_async(synthesize_answer(synth_state))

        assert result["generation"] != ""
        assert "[1]" in result["generation"]
        assert len(result["citations"]) >= 1
        assert result["citations"][0]["source_file"] == "rag_intro.pdf"
        assert len(result["audit_trail"]) == 1

    @patch("core.agents.synthesizer.call_llm_async")
    def test_synthesize_answer_no_documents(self, mock_llm, synth_state):
        """Test synthesis with no documents available."""
        synth_state["relevant_documents"] = []
        synth_state["documents"] = []

        result = run_async(synthesize_answer(synth_state))

        assert "unable to find" in result["generation"].lower()
        assert result["citations"] == []
        mock_llm.assert_not_called()

    @patch("core.agents.synthesizer.call_llm_async")
    def test_synthesize_answer_empty_llm_response(self, mock_llm, synth_state):
        """Test handling of empty LLM response."""
        mock_llm.return_value = ""

        result = run_async(synthesize_answer(synth_state))

        assert result["generation"] != ""  # Should have a fallback message

    @patch("core.agents.synthesizer.call_llm_async")
    def test_synthesize_answer_high_sensitivity_disclaimer(self, mock_llm, synth_state):
        """Test that high sensitivity adds a disclaimer."""
        mock_llm.return_value = "Confidential answer [1]."
        synth_state["relevant_documents"][0]["metadata"]["sensitivity_level"] = "high"

        result = run_async(synthesize_answer(synth_state))

        assert "DISCLAIMER" in result["generation"]


class TestExtractCitations:
    """Tests for the _extract_citations helper."""

    def test_extract_single_citation(self):
        """Test extracting a single citation reference."""
        docs = [
            {
                "doc_id": "d1",
                "text": "Source text",
                "score": 0.9,
                "relevant": True,
                "metadata": {"source_file": "test.pdf", "page_number": 3},
            }
        ]
        response = "The answer is based on [1] research."

        citations = _extract_citations(response, docs)

        assert len(citations) == 1
        assert citations[0]["source_file"] == "test.pdf"
        assert citations[0]["page_number"] == 3

    def test_extract_multiple_citations(self):
        """Test extracting multiple citation references."""
        docs = [
            {
                "doc_id": "d1",
                "text": "Text 1",
                "score": 0.9,
                "relevant": True,
                "metadata": {"source_file": "a.pdf", "page_number": 1},
            },
            {
                "doc_id": "d2",
                "text": "Text 2",
                "score": 0.8,
                "relevant": True,
                "metadata": {"source_file": "b.pdf", "page_number": 2},
            },
            {
                "doc_id": "d3",
                "text": "Text 3",
                "score": 0.7,
                "relevant": True,
                "metadata": {"source_file": "c.pdf", "page_number": 3},
            },
        ]
        response = "Point A [1] and point B [3] are supported."

        citations = _extract_citations(response, docs)

        assert len(citations) == 2
        source_files = [c["source_file"] for c in citations]
        assert "a.pdf" in source_files
        assert "c.pdf" in source_files

    def test_extract_no_citations(self):
        """Test response with no citation markers."""
        docs = [
            {
                "doc_id": "d1",
                "text": "Text",
                "score": 0.9,
                "relevant": True,
                "metadata": {"source_file": "x.pdf", "page_number": 0},
            },
        ]
        response = "This answer has no citations."

        citations = _extract_citations(response, docs)
        assert citations == []

    def test_extract_out_of_range_citation(self):
        """Test that out-of-range citation indices are ignored."""
        docs = [
            {
                "doc_id": "d1",
                "text": "Text",
                "score": 0.9,
                "relevant": True,
                "metadata": {"source_file": "x.pdf", "page_number": 0},
            },
        ]
        response = "See [1] and [5] for details."

        citations = _extract_citations(response, docs)

        # Only [1] should be extracted (index 0), [5] is out of range
        assert len(citations) == 1

    def test_extract_deduplicates_citations(self):
        """Test that repeated citation markers produce unique citations."""
        docs = [
            {
                "doc_id": "d1",
                "text": "Text",
                "score": 0.9,
                "relevant": True,
                "metadata": {"source_file": "x.pdf", "page_number": 0},
            },
        ]
        response = "See [1] and also [1] again."

        citations = _extract_citations(response, docs)
        assert len(citations) == 1


class TestAddDisclaimers:
    """Tests for the _add_disclaimers helper."""

    def test_high_sensitivity_adds_disclaimer(self):
        """Test that high sensitivity adds a strong disclaimer."""
        response = "Some answer."
        result = _add_disclaimers(response, "high")

        assert "DISCLAIMER" in result
        assert "highly sensitive" in result.lower()

    def test_medium_sensitivity_adds_note(self):
        """Test that medium sensitivity adds a note."""
        response = "Some answer."
        result = _add_disclaimers(response, "medium")

        assert "Note" in result
        assert "moderate sensitivity" in result.lower()

    def test_low_sensitivity_no_disclaimer(self):
        """Test that low sensitivity adds no disclaimer."""
        response = "Some answer."
        result = _add_disclaimers(response, "low")

        assert result == response

    def test_disclaimer_appended_not_replacing(self):
        """Test that disclaimers are appended, not replacing content."""
        response = "Original answer content."
        result = _add_disclaimers(response, "high")

        assert result.startswith("Original answer content.")
