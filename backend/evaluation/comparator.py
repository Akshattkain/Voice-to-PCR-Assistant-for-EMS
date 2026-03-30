"""Model comparator — side-by-side evaluation for research hypotheses."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from app.schemas.evaluation import (
    AggregateMetrics,
    BatchComparisonReport,
    SingleEvalResult,
)
from app.schemas.pcr import PCRDocument
from app.services.extraction.base import ExtractionService
from app.utils.logging import logger
from evaluation.metrics import (
    compute_aggregate_f1,
    compute_completeness,
    compute_field_f1,
    compute_hallucination_rate,
)


class ModelComparator:
    """Side-by-side comparison for research hypotheses."""

    def __init__(
        self,
        finetuned_service: Optional[ExtractionService] = None,
        baseline_service: Optional[ExtractionService] = None,
    ):
        self.finetuned = finetuned_service
        self.baseline = baseline_service

    async def evaluate_single(
        self,
        transcript: str,
        ground_truth: PCRDocument,
        extractor: ExtractionService,
    ) -> SingleEvalResult:
        """Evaluate a single (transcript, ground_truth) pair with one extractor."""
        result = await extractor.extract(transcript)

        field_scores = compute_field_f1(result.pcr, ground_truth)
        agg = compute_aggregate_f1(field_scores)
        hallucination = compute_hallucination_rate(result.pcr, ground_truth, transcript)
        completeness = compute_completeness(result.pcr)

        return SingleEvalResult(
            field_scores=field_scores,
            aggregate_f1=agg["macro_f1"],
            hallucination=hallucination,
            completeness=completeness,
            model_used=extractor.model_name,
            latency_ms=result.latency_ms,
        )

    async def compare_batch(
        self,
        dataset_path: str,
    ) -> BatchComparisonReport:
        """Run comparison across entire dataset.

        Expected JSONL format: {"transcript": str, "pcr_json": dict, ...}
        """
        pairs = self._load_dataset(dataset_path)
        logger.info(f"Loaded {len(pairs)} pairs for evaluation")

        ft_results = []
        bl_results = []

        for i, (transcript, gt_pcr) in enumerate(pairs):
            try:
                if self.finetuned:
                    ft_result = await self.evaluate_single(transcript, gt_pcr, self.finetuned)
                    ft_results.append(ft_result)

                if self.baseline:
                    bl_result = await self.evaluate_single(transcript, gt_pcr, self.baseline)
                    bl_results.append(bl_result)

                logger.info(f"Evaluated pair {i+1}/{len(pairs)}")
            except Exception as e:
                logger.error(f"Failed to evaluate pair {i+1}: {e}")

        # Aggregate metrics
        ft_metrics = self._aggregate(ft_results) if ft_results else None
        bl_metrics = self._aggregate(bl_results) if bl_results else None

        # Hypothesis testing
        h1 = self._test_h1(ft_metrics, bl_metrics)
        h2 = self._test_h2(ft_metrics, bl_metrics)
        h3 = self._test_h3(ft_results, bl_results)

        return BatchComparisonReport(
            finetuned_metrics=ft_metrics,
            baseline_metrics=bl_metrics,
            h1_result=h1,
            h2_result=h2,
            h3_result=h3,
            num_samples=len(pairs),
            timestamp=datetime.now(timezone.utc),
        )

    def _load_dataset(self, path: str) -> list[tuple[str, PCRDocument]]:
        """Load evaluation dataset from JSONL file."""
        pairs = []
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                transcript = data["transcript"]
                gt_pcr = PCRDocument(**data["pcr_json"])
                pairs.append((transcript, gt_pcr))
        return pairs

    def _aggregate(self, results: list[SingleEvalResult]) -> AggregateMetrics:
        """Aggregate metrics across evaluation results."""
        if not results:
            return AggregateMetrics(
                micro_f1=0, macro_f1=0, mean_hallucination_rate=0, mean_completeness=0
            )

        f1s = [r.aggregate_f1 for r in results]
        hall_rates = [r.hallucination.hallucination_rate for r in results]
        completeness = [r.completeness.overall_completeness for r in results]

        # Per-field F1 averages
        per_field: dict[str, list[float]] = {}
        for r in results:
            for field, score in r.field_scores.items():
                per_field.setdefault(field, []).append(score.f1)

        per_field_avg = {f: sum(v) / len(v) for f, v in per_field.items()}

        return AggregateMetrics(
            micro_f1=sum(f1s) / len(f1s),
            macro_f1=sum(f1s) / len(f1s),
            mean_hallucination_rate=sum(hall_rates) / len(hall_rates),
            mean_completeness=sum(completeness) / len(completeness),
            per_field_f1=per_field_avg,
        )

    def _test_h1(self, ft: Optional[AggregateMetrics], bl: Optional[AggregateMetrics]) -> str:
        """H1: Fine-tuned F1 > 0.85 and > baseline."""
        if not ft:
            return "NOT_TESTED: No fine-tuned results"
        msg = f"Fine-tuned F1={ft.macro_f1:.3f}"
        if bl:
            msg += f", Baseline F1={bl.macro_f1:.3f}"
        if ft.macro_f1 > 0.85:
            if bl and ft.macro_f1 > bl.macro_f1:
                return f"SUPPORTED: {msg}"
            elif not bl:
                return f"PARTIALLY_SUPPORTED (no baseline): {msg}"
        return f"NOT_SUPPORTED: {msg}"

    def _test_h2(self, ft: Optional[AggregateMetrics], bl: Optional[AggregateMetrics]) -> str:
        """H2: Fine-tuned hallucination rate < 5% and < baseline."""
        if not ft:
            return "NOT_TESTED: No fine-tuned results"
        msg = f"Fine-tuned hallucination={ft.mean_hallucination_rate:.3f}"
        if bl:
            msg += f", Baseline hallucination={bl.mean_hallucination_rate:.3f}"
        if ft.mean_hallucination_rate < 0.05:
            if bl and ft.mean_hallucination_rate < bl.mean_hallucination_rate:
                return f"SUPPORTED: {msg}"
            elif not bl:
                return f"PARTIALLY_SUPPORTED (no baseline): {msg}"
        return f"NOT_SUPPORTED: {msg}"

    def _test_h3(
        self, ft_results: list[SingleEvalResult], bl_results: list[SingleEvalResult]
    ) -> str:
        """H3: Gap detection reduces missing mandatory fields.
        This requires a separate test comparing with/without gap detection pass.
        """
        # H3 is tested via the completeness evaluator, not directly here
        return "DEFERRED: Requires gap detection evaluation (see evaluation/completeness.py)"
