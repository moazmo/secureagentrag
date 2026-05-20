"""Integration tests for the end-to-end RAG pipeline."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from core.graph import create_initial_state
from ingestion.metadata import UserContext


@pytest.fixture()
def mock_user_context() -> UserContext:
    """Create a test user context."""
    return UserContext(
        user_id="test_user",
        org_id="test_org",
        roles=["viewer"],
        clearance_level=1,
    )


@pytest.mark.asyncio
async def test_create_initial_state_structure(mock_user_context: UserContext) -> None:
    """Test that create_initial_state produces a valid GraphState."""
    state = create_initial_state("What is RAG?", mock_user_context)

    assert state["query"] == "What is RAG?"
    assert state["user_context"]["user_id"] == "test_user"
    assert state["query_type"] == ""
    assert state["security_passed"] is False
    assert state["documents"] == []
    assert state["relevant_documents"] == []
    assert state["relevance_ratio"] == 0.0
    assert state["retry_count"] == 0
    assert state["generation"] == ""
    assert state["citations"] == []
    assert state["confidence_score"] == 0.0
    assert state["needs_human_review"] is False
    assert state["evaluation_notes"] == ""
    assert state["audit_trail"] == []


@pytest.mark.asyncio
async def test_async_router_uses_inference_router(mock_user_context: UserContext) -> None:
    """Test that the async router calls call_llm_async which routes through InferenceRouter."""
    from core.agents.router import route_query

    with patch("core.agents.router.call_llm_async") as mock_call:
        mock_call.return_value = "simple"
        state = create_initial_state("test query", mock_user_context)
        result = await route_query(state)

    assert result["query_type"] == "simple"
    mock_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_async_synthesizer_uses_inference_router(mock_user_context: UserContext) -> None:
    """Test that synthesize_answer routes LLM calls through InferenceRouter with sensitivity."""
    from types import SimpleNamespace

    from core.agents.synthesizer import synthesize_answer

    with patch("core.agents.synthesizer.call_llm_with_decision") as mock_call:
        # Synth now expects a (text, decision, response) tuple.
        mock_call.return_value = (
            "Generated answer [1].",
            SimpleNamespace(provider="ollama", model="qwen3:8b", reason="test", forced_local=True),
            SimpleNamespace(text="Generated answer [1].", usage={}, latency_ms=10.0),
        )
        state = create_initial_state("test", mock_user_context)
        state["relevant_documents"] = [
            {
                "doc_id": "d1",
                "text": "Context text",
                "score": 0.9,
                "relevant": True,
                "metadata": {
                    "source_file": "test.pdf",
                    "page_number": 1,
                    "sensitivity_level": "high",
                },
            }
        ]
        result = await synthesize_answer(state)

    assert result["generation"] != ""
    # Verify sensitivity was passed to the LLM call
    call_kwargs = mock_call.call_args.kwargs
    assert call_kwargs.get("sensitivity_level") == "high"


@pytest.mark.asyncio
async def test_async_evaluator_uses_inference_router(mock_user_context: UserContext) -> None:
    """Test that evaluate_response uses async LLM calls."""
    from core.agents.evaluator import evaluate_response

    with patch("core.agents.evaluator.call_llm_async") as mock_call:
        # Evaluator now makes 2 parallel LLM calls (hallucination + completeness)
        mock_call.side_effect = ["NONE", "0.85"]
        state = create_initial_state("test", mock_user_context)
        state["generation"] = "Test answer"
        state["relevant_documents"] = [
            {"doc_id": "d1", "text": "Context", "score": 0.9, "relevant": True, "metadata": {}}
        ]
        result = await evaluate_response(state)

    # Confidence is now computed from multiple metrics, not just LLM parse
    assert result["confidence_score"] > 0.0
    assert isinstance(result["needs_human_review"], bool)
    # Should be called twice (hallucination check + completeness check in parallel)
    assert mock_call.call_count == 2


@pytest.mark.asyncio
async def test_async_grader_uses_inference_router(mock_user_context: UserContext) -> None:
    """Test that grade_documents uses async LLM calls."""
    from core.agents.retriever import grade_documents

    with patch("core.agents.retriever.call_llm_async") as mock_call:
        mock_call.return_value = "yes"
        state = create_initial_state("test", mock_user_context)
        state["documents"] = [
            {
                "doc_id": "d1",
                "text": "Relevant text",
                "score": 0.9,
                "relevant": False,
                "metadata": {},
            }
        ]
        result = await grade_documents(state)

    assert result["relevance_ratio"] == 1.0
    assert len(result["relevant_documents"]) == 1
    mock_call.assert_awaited_once()


@pytest.mark.asyncio
async def test_inference_router_integration() -> None:
    """Test that InferenceRouter.route correctly selects providers based on sensitivity."""
    from inference.router import InferenceRouter

    router = InferenceRouter()

    # HIGH sensitivity should always route to local (ollama)
    decision = router.route("high")
    assert decision.provider == "ollama"
    assert decision.forced_local is True

    # MEDIUM sensitivity should route to local by default
    decision = router.route("medium")
    assert decision.provider == "ollama"

    # LOW sensitivity should route to local by default (no cloud configured)
    decision = router.route("low")
    assert decision.provider == "ollama"


@pytest.mark.asyncio
async def test_sparse_vectors_generated_during_ingestion() -> None:
    """Test that sparse vectors are generated during document ingestion."""
    from retrieval.sparse_embeddings import SparseEmbeddingService

    sparse = SparseEmbeddingService(backend="bm25")
    texts = [
        "the quick brown fox jumps over the lazy dog",
        "the lazy dog sleeps in the sun all day long",
        "a fox is a clever animal that hunts at night",
        "dogs and foxes are both members of the canidae family",
    ]
    vectors = sparse.embed_texts(texts)

    assert len(vectors) == len(texts)
    # At least one vector should have non-empty indices
    assert any(len(v.indices) > 0 for v in vectors)
    # The "canidae" doc should have distinct indices
    canidae_vector = vectors[-1]
    assert len(canidae_vector.indices) > 0


@pytest.mark.asyncio
async def test_txt_loader_integration() -> None:
    """Test that .txt files are supported in the loader."""
    import pathlib
    import tempfile

    from ingestion.loaders import SUPPORTED_EXTENSIONS, load_text

    assert ".txt" in SUPPORTED_EXTENSIONS

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is a test document.\nIt has multiple lines.")
        tmp_path = f.name

    try:
        docs = load_text(tmp_path)
        assert len(docs) == 1
        assert docs[0].file_type == "txt"
        assert "test document" in docs[0].text
    finally:
        pathlib.Path(tmp_path).unlink(missing_ok=True)
