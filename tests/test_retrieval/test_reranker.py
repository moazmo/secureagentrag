"""Tests for the reranker module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from retrieval.hybrid_search import SearchResult


class TestRerankerInit:
    """Tests for Reranker initialization."""

    def test_lazy_initialization(self):
        """Reranker does not load model on init."""
        from retrieval.reranker import Reranker

        reranker = Reranker(model_name="BAAI/bge-reranker-v2-m3")

        assert reranker._model is None

    def test_custom_model_name(self):
        """Reranker accepts custom model name."""
        from retrieval.reranker import Reranker

        reranker = Reranker(model_name="custom/model")

        assert reranker._model_name == "custom/model"

    def test_custom_device(self):
        """Reranker accepts custom device."""
        from retrieval.reranker import Reranker

        reranker = Reranker(device="cpu")

        assert reranker._device == "cpu"


class TestRerankerIsAvailable:
    """Tests for Reranker.is_available()."""

    def test_is_available_returns_bool(self):
        """is_available() returns a boolean."""
        from retrieval.reranker import Reranker

        reranker = Reranker()
        result = reranker.is_available()

        assert isinstance(result, bool)


class TestRerankerPassthrough:
    """Tests for reranker passthrough when model is unavailable."""

    def test_rerank_passthrough_when_unavailable(self):
        """rerank returns documents unchanged when model not available."""
        from retrieval.reranker import Reranker

        reranker = Reranker()

        docs = [
            SearchResult(id="1", text="first", score=0.9),
            SearchResult(id="2", text="second", score=0.8),
            SearchResult(id="3", text="third", score=0.7),
        ]

        with patch.object(reranker, "is_available", return_value=False):
            results = reranker.rerank("test query", docs)

        assert len(results) == 3
        assert results[0].id == "1"
        assert results[1].id == "2"

    def test_rerank_passthrough_with_top_k(self):
        """rerank returns top_k documents when in passthrough mode."""
        from retrieval.reranker import Reranker

        reranker = Reranker()

        docs = [
            SearchResult(id="1", text="first", score=0.9),
            SearchResult(id="2", text="second", score=0.8),
            SearchResult(id="3", text="third", score=0.7),
        ]

        with patch.object(reranker, "is_available", return_value=False):
            results = reranker.rerank("test query", docs, top_k=2)

        assert len(results) == 2

    def test_rerank_empty_documents(self):
        """rerank returns empty list for empty input."""
        from retrieval.reranker import Reranker

        reranker = Reranker()
        results = reranker.rerank("query", [])

        assert results == []

    def test_rerank_texts_passthrough_when_unavailable(self):
        """rerank_texts returns texts with zero scores when unavailable."""
        from retrieval.reranker import Reranker

        reranker = Reranker()
        texts = ["hello world", "foo bar"]

        with patch.object(reranker, "is_available", return_value=False):
            results = reranker.rerank_texts("test", texts)

        assert len(results) == 2
        assert results[0][0] == "hello world"
        assert results[0][1] == 0.0

    def test_rerank_texts_empty(self):
        """rerank_texts returns empty for empty input."""
        from retrieval.reranker import Reranker

        reranker = Reranker()
        results = reranker.rerank_texts("query", [])

        assert results == []


class TestRerankerWithMockedModel:
    """Tests for reranker with mocked cross-encoder model."""

    def test_rerank_sorts_by_score(self):
        """rerank sorts results by cross-encoder scores descending."""
        from retrieval.reranker import Reranker

        reranker = Reranker()

        # Mock the model
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.3, 0.9, 0.6]
        reranker._model = mock_model

        docs = [
            SearchResult(id="1", text="first", score=0.5),
            SearchResult(id="2", text="second", score=0.4),
            SearchResult(id="3", text="third", score=0.3),
        ]

        with patch.object(reranker, "is_available", return_value=True):
            results = reranker.rerank("query", docs)

        # Should be sorted by cross-encoder score: id2(0.9), id3(0.6), id1(0.3)
        assert results[0].id == "2"
        assert results[1].id == "3"
        assert results[2].id == "1"
        assert results[0].score == pytest.approx(0.9)

    def test_rerank_respects_top_k(self):
        """rerank returns only top_k results after scoring."""
        from retrieval.reranker import Reranker

        reranker = Reranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.7, 0.5]
        reranker._model = mock_model

        docs = [
            SearchResult(id="1", text="a", score=0.0),
            SearchResult(id="2", text="b", score=0.0),
            SearchResult(id="3", text="c", score=0.0),
        ]

        with patch.object(reranker, "is_available", return_value=True):
            results = reranker.rerank("query", docs, top_k=2)

        assert len(results) == 2

    def test_rerank_texts_with_model(self):
        """rerank_texts uses model and returns sorted (text, score) tuples."""
        from retrieval.reranker import Reranker

        reranker = Reranker()

        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.8, 0.5]
        reranker._model = mock_model

        texts = ["text a", "text b", "text c"]

        with patch.object(reranker, "is_available", return_value=True):
            results = reranker.rerank_texts("query", texts)

        # Sorted: text_b(0.8), text_c(0.5), text_a(0.2)
        assert results[0] == ("text b", pytest.approx(0.8))
        assert results[1] == ("text c", pytest.approx(0.5))
        assert results[2] == ("text a", pytest.approx(0.2))

    def test_rerank_handles_model_error(self):
        """rerank returns original docs if model prediction fails."""
        from retrieval.reranker import Reranker

        reranker = Reranker()

        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("Model error")
        reranker._model = mock_model

        docs = [
            SearchResult(id="1", text="a", score=0.5),
            SearchResult(id="2", text="b", score=0.3),
        ]

        with patch.object(reranker, "is_available", return_value=True):
            results = reranker.rerank("query", docs)

        # Should fallback to original order
        assert len(results) == 2
        assert results[0].id == "1"
