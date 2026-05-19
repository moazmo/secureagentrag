"""Tests for ColBERT reranker module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from retrieval.colbert_reranker import ColBERTReranker, _torch_cuda


class TestColbertrerankerInit:
    """Tests for ColBERTReranker initialization."""

    def test_unavailable_when_colbert_not_installed(self):
        """If colbert-ai is not installed, is_available() returns False."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", False):
            r = ColBERTReranker()
            assert r.is_available() is False

    def test_available_when_colbert_installed(self):
        """If colbert-ai is installed, is_available() returns True."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", True):
            r = ColBERTReranker()
            assert r.is_available() is True

    def test_default_checkpoint(self):
        """Default checkpoint is colbert-ir/colbertv2.0."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", True):
            r = ColBERTReranker()
            assert r._checkpoint == "colbert-ir/colbertv2.0"

    def test_custom_checkpoint(self):
        """Custom checkpoint is preserved."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", True):
            r = ColBERTReranker(checkpoint="custom/colbert")
            assert r._checkpoint == "custom/colbert"


class TestRerank:
    """Tests for ColBERTReranker.rerank."""

    def test_empty_documents(self):
        """Empty input returns empty list."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", False):
            r = ColBERTReranker()
            assert r.rerank("q", []) == []

    def test_passthrough_when_unavailable(self):
        """When colbert-ai is missing, documents pass through unchanged."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", False):
            r = ColBERTReranker()
            mock_doc = MagicMock()
            mock_doc.text = "doc text"
            result = r.rerank("q", [mock_doc], top_k=1)
            assert result == [mock_doc]

    def test_passthrough_when_index_not_built(self):
        """When index has not been built, documents pass through."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", True):
            r = ColBERTReranker()
            assert r._index_built is False
            mock_doc = MagicMock()
            result = r.rerank("q", [mock_doc])
            assert result == [mock_doc]


class TestRerankTexts:
    """Tests for ColBERTReranker.rerank_texts."""

    def test_empty_texts(self):
        """Empty input returns empty list."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", False):
            r = ColBERTReranker()
            assert r.rerank_texts("q", []) == []

    def test_passthrough_when_unavailable(self):
        """When unavailable, returns texts with zero scores."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", False):
            r = ColBERTReranker()
            result = r.rerank_texts("q", ["a", "b"])
            assert result == [("a", 0.0), ("b", 0.0)]

    def test_top_k_truncation(self):
        """top_k limits the number of returned results."""
        with patch("retrieval.colbert_reranker._COLBERT_AVAILABLE", False):
            r = ColBERTReranker()
            result = r.rerank_texts("q", ["a", "b", "c"], top_k=2)
            assert len(result) == 2


class TestTorchCuda:
    """Tests for _torch_cuda helper."""

    def test_returns_false_when_torch_missing(self):
        """If torch is not installed, returns False."""
        with patch.dict("sys.modules", {"torch": None}):
            assert _torch_cuda() is False

    def test_returns_true_when_cuda_available(self):
        """When torch.cuda.is_available() is True, returns True."""
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            assert _torch_cuda() is True
