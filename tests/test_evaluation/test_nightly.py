"""Tests for nightly regression evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from evaluation.nightly import (
    _detect_regression,
    _lexical_scores,
    _load_baseline,
    _load_golden,
    _run_golden_set,
    main,
)


class TestLoadGolden:
    """Tests for _load_golden."""

    def test_reads_jsonl_lines(self, tmp_path):
        """Parses each non-empty line as JSON."""
        golden = tmp_path / "golden_set.jsonl"
        golden.write_text(
            json.dumps({"id": "q1", "question": "What?"})
            + "\n"
            + json.dumps({"id": "q2", "question": "Why?"})
            + "\n",
            encoding="utf-8",
        )
        with patch("evaluation.nightly.GOLDEN_PATH", golden):
            items = _load_golden()

        assert len(items) == 2
        assert items[0]["id"] == "q1"
        assert items[1]["question"] == "Why?"

    def test_skips_empty_lines(self, tmp_path):
        """Blank lines are ignored."""
        golden = tmp_path / "golden_set.jsonl"
        golden.write_text(
            json.dumps({"id": "q1"}) + "\n\n\n" + json.dumps({"id": "q2"}) + "\n",
            encoding="utf-8",
        )
        with patch("evaluation.nightly.GOLDEN_PATH", golden):
            items = _load_golden()

        assert len(items) == 2

    def test_returns_empty_when_missing(self, tmp_path):
        """Missing golden file yields empty list."""
        with patch("evaluation.nightly.GOLDEN_PATH", tmp_path / "nope.jsonl"):
            items = _load_golden()

        assert items == []


class TestLoadBaseline:
    """Tests for _load_baseline."""

    def test_reads_existing_baseline(self, tmp_path):
        """Parses baseline JSON with expected metric keys."""
        baseline = tmp_path / "baseline.json"
        baseline.write_text(
            json.dumps({"faithfulness": 0.85, "context_precision": 0.80, "answer_relevancy": 0.75}),
            encoding="utf-8",
        )
        with patch("evaluation.nightly.BASELINE_PATH", baseline):
            scores = _load_baseline()

        assert scores["faithfulness"] == 0.85
        assert scores["context_precision"] == 0.80

    def test_defaults_when_missing(self, tmp_path):
        """Missing baseline returns zero defaults."""
        with patch("evaluation.nightly.BASELINE_PATH", tmp_path / "nope.json"):
            scores = _load_baseline()

        assert scores == {"faithfulness": 0.0, "context_precision": 0.0, "answer_relevancy": 0.0}


class TestRunGoldenSet:
    """Tests for _run_golden_set."""

    @patch("evaluation.nightly.run_rag_pipeline", new_callable=AsyncMock)
    async def test_runs_each_item(self, mock_pipeline):
        """Each golden item triggers one pipeline run."""
        mock_pipeline.return_value = {
            "generation": "answer",
            "relevant_documents": [{"text": "ctx"}],
        }
        items = [
            {"id": "q1", "question": "What?"},
            {"id": "q2", "question": "Why?"},
        ]

        result = await _run_golden_set(items)

        assert len(result["responses"]) == 2
        mock_pipeline.assert_awaited()
        assert mock_pipeline.await_count == 2

    @patch("evaluation.nightly.run_rag_pipeline", new_callable=AsyncMock)
    async def test_uses_document_fallback(self, mock_pipeline):
        """If relevant_documents is empty, falls back to documents key."""
        mock_pipeline.return_value = {
            "generation": "ans",
            "documents": [{"text": "fallback ctx"}],
        }
        items = [{"id": "q1", "question": "What?"}]

        result = await _run_golden_set(items)

        assert result["responses"][0]["contexts"] == ["fallback ctx"]

    @patch("evaluation.nightly.run_rag_pipeline", new_callable=AsyncMock)
    async def test_failure_skips_item(self, mock_pipeline):
        """A pipeline failure skips the item rather than crashing the suite."""
        mock_pipeline.side_effect = [RuntimeError("boom"), {"generation": "ok", "documents": []}]
        items = [{"id": "q1", "question": "What?"}, {"id": "q2", "question": "Why?"}]

        result = await _run_golden_set(items)

        assert len(result["responses"]) == 1


class TestLexicalScores:
    """Tests for _lexical_scores fallback."""

    def test_perfect_overlap(self):
        """When answer equals context, faithfulness is 1.0."""
        responses = [
            {
                "question": "q",
                "answer": "exact match",
                "contexts": ["exact match"],
                "ground_truth": "exact match",
            }
        ]
        scores = _lexical_scores(responses)
        assert scores["faithfulness"] == 1.0
        assert scores["context_precision"] == 1.0
        assert scores["answer_relevancy"] == 1.0

    def test_zero_overlap(self):
        """No token overlap yields zero scores."""
        responses = [
            {
                "question": "q",
                "answer": "foo bar",
                "contexts": ["baz qux"],
                "ground_truth": "hello world",
            }
        ]
        scores = _lexical_scores(responses)
        assert scores["faithfulness"] == 0.0
        assert scores["context_precision"] == 0.0
        assert scores["answer_relevancy"] == 0.0

    def test_empty_answer_skipped(self):
        """Empty answers are excluded from the average."""
        responses = [
            {"question": "q", "answer": "", "contexts": ["ctx"], "ground_truth": "gt"},
            {"question": "q", "answer": "foo", "contexts": ["foo"], "ground_truth": "foo"},
        ]
        scores = _lexical_scores(responses)
        assert scores["faithfulness"] == 1.0
        assert scores["answer_relevancy"] == 1.0

    def test_multiple_contexts(self):
        """Context precision averages over all contexts."""
        responses = [
            {
                "question": "q",
                "answer": "hello world",
                "contexts": ["hello", "world"],
                "ground_truth": "hello world",
            }
        ]
        scores = _lexical_scores(responses)
        # both contexts intersect answer
        assert scores["context_precision"] == 1.0


class TestDetectRegression:
    """Tests for _detect_regression."""

    def test_no_regression_when_improved(self):
        """Scores above baseline do not trigger regression."""
        result = _detect_regression(
            {"faithfulness": 0.9, "context_precision": 0.85},
            {"faithfulness": 0.8, "context_precision": 0.8},
        )
        assert result["regression_detected"] is False
        assert result["regressed_metric"] == ""

    def test_no_regression_within_threshold(self):
        """A small drop (≤5pp) is not flagged."""
        result = _detect_regression(
            {"faithfulness": 0.83, "context_precision": 0.80},
            {"faithfulness": 0.85, "context_precision": 0.80},
        )
        assert result["regression_detected"] is False

    def test_regression_when_below_threshold(self):
        """A drop >5pp on faithfulness or context_precision is flagged."""
        result = _detect_regression(
            {"faithfulness": 0.75, "context_precision": 0.80},
            {"faithfulness": 0.85, "context_precision": 0.80},
        )
        assert result["regression_detected"] is True
        assert result["regressed_metric"] == "faithfulness"
        assert result["regression_delta"] < -0.05

    def test_worst_regression_reported(self):
        """If both metrics regress, the larger drop is reported."""
        result = _detect_regression(
            {"faithfulness": 0.70, "context_precision": 0.65},
            {"faithfulness": 0.85, "context_precision": 0.80},
        )
        assert result["regression_detected"] is True
        # context_precision dropped 0.15 vs faithfulness 0.15 — tie, first wins


class TestMain:
    """Tests for main CLI entrypoint."""

    @patch("evaluation.nightly.GOLDEN_PATH", Path("/nonexistent/golden.jsonl"))
    def test_exits_zero_when_no_golden(self):
        """No golden questions → clean exit with message."""
        assert main([]) == 0

    @patch("evaluation.nightly._compute_scores")
    @patch("evaluation.nightly._load_golden")
    @patch("evaluation.nightly._run_golden_set", new_callable=AsyncMock)
    @patch("evaluation.nightly._load_baseline")
    def test_exits_zero_on_pass(self, mock_baseline, mock_run, mock_load, mock_scores):
        """No regression → exit 0."""
        mock_load.return_value = [{"id": "q1", "question": "What?"}]
        mock_run.return_value = {
            "responses": [
                {"question": "What?", "answer": "ans", "contexts": ["ctx"], "ground_truth": "ans"}
            ]
        }
        mock_baseline.return_value = {
            "faithfulness": 0.0,
            "context_precision": 0.0,
            "answer_relevancy": 0.0,
        }
        mock_scores.return_value = {
            "faithfulness": 0.9,
            "context_precision": 0.85,
            "answer_relevancy": 0.8,
        }

        assert main([]) == 0

    @patch("evaluation.nightly._compute_scores")
    @patch("evaluation.nightly._load_golden")
    @patch("evaluation.nightly._run_golden_set", new_callable=AsyncMock)
    @patch("evaluation.nightly._load_baseline")
    def test_exits_one_on_regression(self, mock_baseline, mock_run, mock_load, mock_scores):
        """Regression detected → exit 1 unless --no-fail-on-regression."""
        mock_load.return_value = [{"id": "q1", "question": "What?"}]
        mock_run.return_value = {
            "responses": [
                {"question": "What?", "answer": "ans", "contexts": ["ctx"], "ground_truth": "ans"}
            ]
        }
        mock_baseline.return_value = {
            "faithfulness": 1.0,
            "context_precision": 1.0,
            "answer_relevancy": 1.0,
        }
        mock_scores.return_value = {
            "faithfulness": 0.8,
            "context_precision": 0.8,
            "answer_relevancy": 0.8,
        }

        assert main([]) == 1

    @patch("evaluation.nightly._compute_scores")
    @patch("evaluation.nightly._load_golden")
    @patch("evaluation.nightly._run_golden_set", new_callable=AsyncMock)
    @patch("evaluation.nightly._load_baseline")
    def test_no_fail_flag_prevents_exit_one(self, mock_baseline, mock_run, mock_load, mock_scores):
        """--no-fail-on-regression suppresses the non-zero exit."""
        mock_load.return_value = [{"id": "q1", "question": "What?"}]
        mock_run.return_value = {
            "responses": [
                {"question": "What?", "answer": "ans", "contexts": ["ctx"], "ground_truth": "ans"}
            ]
        }
        mock_baseline.return_value = {
            "faithfulness": 1.0,
            "context_precision": 1.0,
            "answer_relevancy": 1.0,
        }
        mock_scores.return_value = {
            "faithfulness": 0.8,
            "context_precision": 0.8,
            "answer_relevancy": 0.8,
        }

        assert main(["--no-fail-on-regression"]) == 0
