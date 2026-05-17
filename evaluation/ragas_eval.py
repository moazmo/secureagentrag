"""RAG evaluation pipeline using Ragas metrics.

Provides automated evaluation of RAG pipeline quality using Ragas
metrics (faithfulness, answer relevancy, context precision, context recall).
Gracefully handles missing ragas dependency — returns None scores when unavailable.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from config.settings import settings
from utils.logging import get_logger

logger = get_logger(__name__)

# Conditional ragas import
try:
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    _RAGAS_AVAILABLE = True
except ImportError:
    _RAGAS_AVAILABLE = False
    logger.info("ragas_not_installed", msg="Evaluation will return None scores")


class EvalSample(BaseModel):
    """A single evaluation sample with query, response, and contexts.

    Attributes:
        query: The user's original query.
        response: The generated response from the RAG pipeline.
        contexts: List of context strings used to generate the response.
        ground_truth: Optional reference answer for recall metrics.
    """

    query: str
    response: str
    contexts: list[str]
    ground_truth: str | None = None


class EvalResult(BaseModel):
    """Evaluation result for a single sample.

    Attributes:
        sample: The evaluated sample.
        faithfulness: Score measuring if response is grounded in contexts (0-1).
        answer_relevancy: Score measuring response relevance to query (0-1).
        context_precision: Score measuring precision of retrieved contexts (0-1).
        context_recall: Score measuring recall of relevant information (0-1).
        overall_score: Weighted average of all available metrics.
        timestamp: When the evaluation was performed.
        model: Model used for evaluation judgments.
        latency_ms: Time taken for evaluation in milliseconds.
    """

    sample: EvalSample
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    overall_score: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    model: str = ""
    latency_ms: float = 0.0


class RagasEvaluator:
    """Evaluates RAG pipeline quality using Ragas metrics.

    Provides async evaluation of individual samples or batches.
    Falls back gracefully when ragas is not installed.

    Args:
        llm_provider: LLM provider to use for evaluation judgments.
    """

    def __init__(self, llm_provider: str = "ollama") -> None:
        """Initialize the Ragas evaluator.

        Args:
            llm_provider: Provider for the evaluation LLM (default: "ollama").
        """
        self._provider = llm_provider
        self._model = settings.llm_model

    def is_available(self) -> bool:
        """Check if the ragas library is importable and ready.

        Returns:
            True if ragas is installed and available, False otherwise.
        """
        return _RAGAS_AVAILABLE

    async def evaluate_single(self, sample: EvalSample) -> EvalResult:
        """Run Ragas metrics on a single evaluation sample.

        If ragas is not installed, returns an EvalResult with all
        metric scores set to None.

        Args:
            sample: The EvalSample to evaluate.

        Returns:
            EvalResult with computed metric scores (or None if unavailable).
        """
        start_time = time.perf_counter()

        if not _RAGAS_AVAILABLE:
            logger.warning("ragas_evaluation_skipped", reason="ragas not installed")
            return EvalResult(
                sample=sample,
                model=self._model,
                latency_ms=0.0,
            )

        try:
            # Build dataset for ragas
            from datasets import Dataset

            eval_data = {
                "question": [sample.query],
                "answer": [sample.response],
                "contexts": [sample.contexts],
            }
            if sample.ground_truth:
                eval_data["ground_truth"] = [sample.ground_truth]

            dataset = Dataset.from_dict(eval_data)

            # Select metrics based on available data
            metrics = [faithfulness, answer_relevancy, context_precision]
            if sample.ground_truth:
                metrics.append(context_recall)

            # Run evaluation
            result = ragas_evaluate(dataset=dataset, metrics=metrics)
            result_df = result.to_pandas()

            elapsed_ms = (time.perf_counter() - start_time) * 1000

            # Extract scores
            faith_score = self._safe_get_score(result_df, "faithfulness")
            relevancy_score = self._safe_get_score(result_df, "answer_relevancy")
            precision_score = self._safe_get_score(result_df, "context_precision")
            recall_score = self._safe_get_score(result_df, "context_recall")

            # Compute overall score (average of non-None scores)
            scores = [
                s
                for s in [faith_score, relevancy_score, precision_score, recall_score]
                if s is not None
            ]
            overall = sum(scores) / len(scores) if scores else None

            return EvalResult(
                sample=sample,
                faithfulness=faith_score,
                answer_relevancy=relevancy_score,
                context_precision=precision_score,
                context_recall=recall_score,
                overall_score=overall,
                model=self._model,
                latency_ms=elapsed_ms,
            )

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error("ragas_evaluation_failed", error=str(exc))
            return EvalResult(
                sample=sample,
                model=self._model,
                latency_ms=elapsed_ms,
            )

    async def evaluate_batch(self, samples: list[EvalSample]) -> list[EvalResult]:
        """Evaluate multiple samples sequentially.

        Args:
            samples: List of EvalSample objects to evaluate.

        Returns:
            List of EvalResult objects, one per sample.
        """
        results: list[EvalResult] = []
        logger.info("batch_evaluation_started", count=len(samples))

        for i, sample in enumerate(samples):
            logger.debug("evaluating_sample", index=i, total=len(samples))
            result = await self.evaluate_single(sample)
            results.append(result)

        successful = sum(1 for r in results if r.overall_score is not None)
        logger.info(
            "batch_evaluation_completed",
            total=len(results),
            successful=successful,
        )
        return results

    @staticmethod
    def _safe_get_score(df: Any, column: str) -> float | None:
        """Safely extract a score from the ragas result DataFrame.

        Args:
            df: Pandas DataFrame from ragas evaluation.
            column: Column name to extract.

        Returns:
            Float score value, or None if column doesn't exist or is NaN.
        """
        try:
            if column in df.columns:
                value = df[column].iloc[0]
                if value is not None and not (isinstance(value, float) and value != value):
                    return float(value)
        except (IndexError, KeyError, TypeError):
            pass
        return None
