"""Tests for evaluation benchmark module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from evaluation.benchmark import (
    _ensure_corpus,
    _print_results,
    _run_single_benchmark,
    _warmup,
    run_benchmarks,
)


class TestEnsureCorpus:
    """Tests for _ensure_corpus."""

    @patch("retrieval.qdrant_client.QdrantManager")
    def test_returns_true_when_documents_exist(self, mock_mgr_cls):
        """If Qdrant reports >0 documents, corpus exists."""
        mock_mgr = MagicMock()
        mock_mgr.get_document_count.return_value = 147
        mock_mgr_cls.return_value = mock_mgr

        assert _ensure_corpus() is True

    @patch("retrieval.qdrant_client.QdrantManager")
    def test_returns_false_when_empty(self, mock_mgr_cls):
        """Zero documents means no corpus."""
        mock_mgr = MagicMock()
        mock_mgr.get_document_count.return_value = 0
        mock_mgr_cls.return_value = mock_mgr

        assert _ensure_corpus() is False

    @patch("retrieval.qdrant_client.QdrantManager")
    def test_returns_false_on_exception(self, mock_mgr_cls):
        """Qdrant unreachable is treated as no corpus."""
        mock_mgr_cls.side_effect = RuntimeError("connection refused")

        assert _ensure_corpus() is False


class TestWarmup:
    """Tests for _warmup."""

    @patch("evaluation.benchmark.run_rag_pipeline", new_callable=AsyncMock)
    async def test_runs_warmup_query(self, mock_pipeline):
        """Warmup executes one RAG pipeline call."""
        mock_pipeline.return_value = {}

        await _warmup()

        mock_pipeline.assert_awaited_once()
        kwargs = mock_pipeline.call_args.kwargs
        assert kwargs["query"] == "What is this document about?"

    @patch("evaluation.benchmark.run_rag_pipeline", new_callable=AsyncMock)
    async def test_logs_failure_but_does_not_raise(self, mock_pipeline):
        """A failed warmup is logged, not raised."""
        mock_pipeline.side_effect = RuntimeError("timeout")

        await _warmup()  # should not raise


class TestRunSingleBenchmark:
    """Tests for _run_single_benchmark."""

    @patch("evaluation.benchmark.run_rag_pipeline", new_callable=AsyncMock)
    async def test_collects_metrics_on_success(self, mock_pipeline):
        """A successful run populates all expected metric fields."""
        mock_pipeline.return_value = {
            "security_passed": True,
            "relevance_ratio": 0.8,
            "retry_count": 1,
            "confidence_score": 0.95,
            "citations": [{}, {}],
            "generation": "The generated answer.",
        }

        result = await _run_single_benchmark("What is RAG?", "simple")

        assert result["query"] == "What is RAG?"
        assert result["query_type"] == "simple"
        assert result["success"] is True
        assert result["security_passed"] is True
        assert result["relevance_ratio"] == 0.8
        assert result["retry_count"] == 1
        assert result["confidence_score"] == 0.95
        assert result["citation_count"] == 2
        assert result["generation_length"] == len("The generated answer.")
        assert result["latency_ms"] > 0

    @patch("evaluation.benchmark.run_rag_pipeline", new_callable=AsyncMock)
    async def test_failure_records_error(self, mock_pipeline):
        """A pipeline exception marks success=False and captures the error."""
        mock_pipeline.side_effect = RuntimeError("pipeline crash")

        result = await _run_single_benchmark("bad query", "complex")

        assert result["success"] is False
        assert "pipeline crash" in result["error"]
        assert result["latency_ms"] >= 0


class TestRunBenchmarks:
    """Tests for run_benchmarks orchestrator."""

    @patch("evaluation.benchmark._ensure_corpus")
    @patch("evaluation.benchmark._warmup")
    @patch("evaluation.benchmark.run_rag_pipeline", new_callable=AsyncMock)
    async def test_short_circuits_when_no_corpus(self, mock_pipeline, mock_warmup, mock_corpus):
        """Without ingested documents the suite returns an error dict."""
        mock_corpus.return_value = False

        result = await run_benchmarks()

        assert "error" in result
        mock_warmup.assert_not_called()
        mock_pipeline.assert_not_called()

    @patch("evaluation.benchmark._ensure_corpus")
    @patch("evaluation.benchmark._warmup")
    @patch("evaluation.benchmark.run_rag_pipeline", new_callable=AsyncMock)
    async def test_runs_all_query_types(self, mock_pipeline, mock_warmup, mock_corpus):
        """The suite runs simple, complex, and arabic query buckets."""
        mock_corpus.return_value = True
        mock_pipeline.return_value = {
            "security_passed": True,
            "relevance_ratio": 0.5,
            "retry_count": 0,
            "confidence_score": 0.8,
            "citations": [{}],
            "generation": "ans",
        }

        result = await run_benchmarks(runs_per_type=1)

        assert "timestamp" in result
        assert "configuration" in result
        assert "summary" in result
        assert "raw_results" in result
        # Should have 1 query per type = 3 total raw results
        assert len(result["raw_results"]) == 3
        for qt in ("simple", "complex", "arabic"):
            assert qt in result["summary"]
            stats = result["summary"][qt]
            assert stats["count"] == 1
            assert "mean_ms" in stats
            assert "avg_confidence" in stats

    @patch("evaluation.benchmark._ensure_corpus")
    @patch("evaluation.benchmark._warmup")
    @patch("evaluation.benchmark.run_rag_pipeline", new_callable=AsyncMock)
    async def test_partial_failure_in_bucket(self, mock_pipeline, mock_warmup, mock_corpus):
        """If some queries in a bucket fail, stats use only successful ones."""
        mock_corpus.return_value = True
        mock_pipeline.side_effect = [
            {
                "security_passed": True,
                "relevance_ratio": 0.5,
                "retry_count": 0,
                "confidence_score": 0.8,
                "citations": [{}],
                "generation": "a",
            },
            RuntimeError("fail"),
        ]

        result = await run_benchmarks(runs_per_type=2)

        # With runs_per_type=2, each bucket gets 2 calls. Simple bucket: 1 success, 1 fail.
        simple_stats = result["summary"]["simple"]
        assert simple_stats["count"] == 1


class TestPrintResults:
    """Tests for _print_results formatting."""

    def test_prints_summary_table(self, capsys):
        """Output contains headers and per-type rows."""
        report = {
            "timestamp": "2026-01-01T00:00:00",
            "configuration": {
                "model": "qwen3:8b",
                "embedding_model": "bge-m3",
            },
            "summary": {
                "simple": {
                    "count": 3,
                    "mean_ms": 1000.0,
                    "median_ms": 950.0,
                    "min_ms": 900.0,
                    "max_ms": 1100.0,
                    "p90_ms": 1100.0,
                    "stddev_ms": 50.0,
                    "avg_relevance_ratio": 0.8,
                    "avg_confidence": 0.9,
                },
            },
        }

        _print_results(report)
        captured = capsys.readouterr()

        assert "SECUREAGENTRAG BENCHMARK RESULTS" in captured.out
        assert "simple" in captured.out
        assert "1000ms" in captured.out
        assert "relevance=0.80" in captured.out

    def test_handles_error_bucket(self, capsys):
        """A bucket with all failures prints the error message."""
        report = {
            "timestamp": "2026-01-01T00:00:00",
            "configuration": {"model": "qwen3:8b", "embedding_model": "bge-m3"},
            "summary": {"complex": {"error": "All queries failed"}},
        }

        _print_results(report)
        captured = capsys.readouterr()

        assert "All queries failed" in captured.out
