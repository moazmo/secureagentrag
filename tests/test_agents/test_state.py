"""Tests for the GraphState schema and TypedDict definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.state import Citation, DocumentGrade, GraphState


class TestDocumentGrade:
    """Tests for the DocumentGrade TypedDict."""

    def test_create_document_grade(self):
        """Test creating a DocumentGrade with all required fields."""
        doc: DocumentGrade = {
            "doc_id": "test-123",
            "text": "Sample document text.",
            "score": 0.85,
            "relevant": True,
            "metadata": {"source_file": "test.pdf", "page_number": 1},
        }
        assert doc["doc_id"] == "test-123"
        assert doc["text"] == "Sample document text."
        assert doc["score"] == 0.85
        assert doc["relevant"] is True
        assert doc["metadata"]["source_file"] == "test.pdf"

    def test_document_grade_with_empty_metadata(self):
        """Test DocumentGrade with empty metadata dict."""
        doc: DocumentGrade = {
            "doc_id": "empty-meta",
            "text": "Some text",
            "score": 0.0,
            "relevant": False,
            "metadata": {},
        }
        assert doc["metadata"] == {}


class TestCitation:
    """Tests for the Citation TypedDict."""

    def test_create_citation(self):
        """Test creating a Citation with all required fields."""
        citation: Citation = {
            "source_file": "report.pdf",
            "page_number": 5,
            "chunk_text": "Key finding from the report...",
            "relevance_score": 0.92,
        }
        assert citation["source_file"] == "report.pdf"
        assert citation["page_number"] == 5
        assert citation["chunk_text"] == "Key finding from the report..."
        assert citation["relevance_score"] == 0.92

    def test_citation_with_zero_page(self):
        """Test Citation with page_number 0."""
        citation: Citation = {
            "source_file": "doc.pdf",
            "page_number": 0,
            "chunk_text": "First page content",
            "relevance_score": 0.5,
        }
        assert citation["page_number"] == 0


class TestGraphState:
    """Tests for the GraphState TypedDict."""

    def test_create_full_graph_state(self):
        """Test creating a GraphState with all fields populated."""
        state: GraphState = {
            "query": "What is RAG?",
            "user_context": {
                "user_id": "user1",
                "org_id": "org1",
                "roles": ["admin"],
                "clearance_level": 3,
            },
            "query_type": "simple",
            "rewritten_query": "What is Retrieval-Augmented Generation?",
            "security_passed": True,
            "security_message": "Passed",
            "documents": [],
            "relevant_documents": [],
            "relevance_ratio": 0.8,
            "retry_count": 0,
            "max_retries": 2,
            "generation": "RAG is a technique...",
            "citations": [],
            "confidence_score": 0.9,
            "needs_human_review": False,
            "evaluation_notes": "High confidence.",
            "audit_trail": [],
        }
        assert state["query"] == "What is RAG?"
        assert state["security_passed"] is True
        assert state["confidence_score"] == 0.9

    def test_audit_trail_append_behavior(self):
        """Test that audit_trail supports list append semantics (reducer pattern)."""
        # Simulate how LangGraph uses the Annotated[list[dict], add] reducer
        from operator import add

        trail_a = [{"node": "router", "action": "route"}]
        trail_b = [{"node": "security", "action": "check"}]

        # The `add` operator concatenates lists
        merged = add(trail_a, trail_b)
        assert len(merged) == 2
        assert merged[0]["node"] == "router"
        assert merged[1]["node"] == "security"

    def test_audit_trail_multiple_appends(self):
        """Test multiple sequential appends to audit_trail via reducer."""
        from operator import add

        trail = []
        trail = add(trail, [{"node": "router"}])
        trail = add(trail, [{"node": "security"}])
        trail = add(trail, [{"node": "retriever"}])

        assert len(trail) == 3
        assert [entry["node"] for entry in trail] == [
            "router",
            "security",
            "retriever",
        ]

    def test_graph_state_with_documents(self):
        """Test GraphState with populated documents list."""
        doc: DocumentGrade = {
            "doc_id": "d1",
            "text": "Test text",
            "score": 0.9,
            "relevant": True,
            "metadata": {"source_file": "test.pdf"},
        }
        state: GraphState = {
            "query": "test",
            "user_context": {"user_id": "u1", "org_id": "o1", "roles": [], "clearance_level": 1},
            "query_type": "simple",
            "rewritten_query": "",
            "security_passed": True,
            "security_message": "",
            "documents": [doc],
            "relevant_documents": [doc],
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
        assert len(state["documents"]) == 1
        assert state["documents"][0]["doc_id"] == "d1"
