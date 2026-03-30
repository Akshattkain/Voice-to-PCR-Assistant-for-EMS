"""CLI entry point for batch evaluation.

Usage:
    python -m evaluation.run_evaluation \
        --dataset training/data/processed/test.jsonl \
        --models llm_baseline \
        --output evaluation/reports/
"""

import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path

from app.core.gap_detector import GapDetector
from app.dependencies import get_extraction_service
from app.utils.logging import logger
from evaluation.comparator import ModelComparator
from evaluation.completeness import GapDetectionEvaluator


async def run(args):
    """Run batch evaluation."""
    # Initialize extractors
    finetuned = None
    baseline = None

    if "finetuned" in args.models:
        try:
            finetuned = get_extraction_service("finetuned")
            logger.info("Loaded fine-tuned extractor")
        except Exception as e:
            logger.warning(f"Could not load fine-tuned extractor: {e}")

    if "llm_baseline" in args.models:
        baseline = get_extraction_service("llm_baseline")
        logger.info("Loaded LLM baseline extractor")

    # Run model comparison
    comparator = ModelComparator(
        finetuned_service=finetuned,
        baseline_service=baseline,
    )

    logger.info(f"Running batch comparison on {args.dataset}")
    report = await comparator.compare_batch(args.dataset)

    # Run gap detection evaluation
    gap_report = None
    extractor = finetuned or baseline
    if extractor:
        evaluator = GapDetectionEvaluator(GapDetector())
        gap_report = await evaluator.evaluate(args.dataset, extractor)

    # Save reports
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = output_dir / f"comparison_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report.model_dump(mode="json"), f, indent=2, default=str)
    logger.info(f"Comparison report saved to {report_path}")

    if gap_report:
        gap_path = output_dir / f"gap_detection_report_{timestamp}.json"
        with open(gap_path, "w") as f:
            json.dump(gap_report, f, indent=2)
        logger.info(f"Gap detection report saved to {gap_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Samples evaluated: {report.num_samples}")

    if report.finetuned_metrics:
        m = report.finetuned_metrics
        print(f"\nFine-tuned Model:")
        print(f"  Macro F1: {m.macro_f1:.3f}")
        print(f"  Hallucination Rate: {m.mean_hallucination_rate:.3f}")
        print(f"  Completeness: {m.mean_completeness:.3f}")

    if report.baseline_metrics:
        m = report.baseline_metrics
        print(f"\nLLM Baseline:")
        print(f"  Macro F1: {m.macro_f1:.3f}")
        print(f"  Hallucination Rate: {m.mean_hallucination_rate:.3f}")
        print(f"  Completeness: {m.mean_completeness:.3f}")

    print(f"\nH1 (F1 > 0.85): {report.h1_result}")
    print(f"H2 (Hallucination < 5%): {report.h2_result}")
    print(f"H3 (Gap detection): {report.h3_result}")

    if gap_report:
        print(f"\nGap Detection Evaluation:")
        print(f"  Completeness without: {gap_report['avg_completeness_without_gap_detection']:.3f}")
        print(f"  Completeness with: {gap_report['avg_completeness_with_gap_detection']:.3f}")
        print(f"  H3: {gap_report['h3_result']}")

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run batch evaluation")
    parser.add_argument(
        "--dataset", required=True, help="Path to test JSONL file"
    )
    parser.add_argument(
        "--models", nargs="+", default=["llm_baseline"],
        choices=["finetuned", "llm_baseline"],
        help="Models to evaluate",
    )
    parser.add_argument(
        "--output", default="evaluation/reports/",
        help="Output directory for reports",
    )
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
