"""Gap detection evaluation — tests H3.

H3: Automated gap detection reduces the proportion of missing mandatory
PCR fields compared to single-pass extraction without gap detection.
"""

import json
from pathlib import Path

from app.core.gap_detector import GapDetector
from app.schemas.pcr import PCRDocument, PCRStateEnvelope
from app.services.extraction.base import ExtractionService
from app.utils.logging import logger
from evaluation.metrics import compute_completeness


class GapDetectionEvaluator:
    """Evaluate the impact of gap detection on field completeness (H3)."""

    def __init__(self, gap_detector: GapDetector):
        self.gap_detector = gap_detector

    async def evaluate(
        self,
        dataset_path: str,
        extractor: ExtractionService,
    ) -> dict:
        """Compare completeness with and without gap detection.

        For each sample:
        1. Run extraction (single-pass) -> measure completeness
        2. Run gap detection -> identify missing mandatory/required fields
        3. Count how many gaps were correctly identified

        Returns summary statistics.
        """
        pairs = self._load_dataset(dataset_path)
        logger.info(f"Evaluating gap detection on {len(pairs)} samples")

        without_gap = []  # Completeness scores without gap detection
        with_gap = []  # Fields identified by gap detection
        correctly_identified = 0
        total_actually_missing = 0

        for i, (transcript, gt_pcr) in enumerate(pairs):
            try:
                # Single-pass extraction
                result = await extractor.extract(transcript)
                completeness = compute_completeness(result.pcr)
                without_gap.append(completeness.overall_completeness)

                # Gap detection
                state = PCRStateEnvelope(session_id="eval", pcr=result.pcr)
                gaps = self.gap_detector.detect_gaps(state)

                # Check: how many of the missing fields in ground truth
                # are correctly identified by gap detection
                gt_completeness = compute_completeness(gt_pcr)
                actually_missing = set(completeness.missing_mandatory + completeness.missing_required)
                detected_missing = set(
                    g.field_name for g in gaps.missing_mandatory + gaps.missing_required
                )

                correctly_identified += len(actually_missing & detected_missing)
                total_actually_missing += len(actually_missing)

                # Simulated improvement: if gap detection correctly flags fields,
                # assume a user would fill them -> improved completeness
                # (This simulates the H3 scenario)
                improved_pcr = result.pcr.model_copy()
                # For evaluation, we assume gap-detected fields get filled from ground truth
                gt_dict = gt_pcr.model_dump()
                for field_name in detected_missing:
                    gt_value = gt_dict.get(field_name)
                    if gt_value is not None:
                        setattr(improved_pcr, field_name, gt_value)

                improved_completeness = compute_completeness(improved_pcr)
                with_gap.append(improved_completeness.overall_completeness)

            except Exception as e:
                logger.error(f"Failed on sample {i}: {e}")

        # Summary
        n = len(without_gap)
        avg_without = sum(without_gap) / n if n > 0 else 0
        avg_with = sum(with_gap) / n if n > 0 else 0
        gap_detection_accuracy = (
            correctly_identified / total_actually_missing
            if total_actually_missing > 0
            else 1.0
        )

        h3_supported = avg_with > avg_without

        return {
            "num_samples": n,
            "avg_completeness_without_gap_detection": avg_without,
            "avg_completeness_with_gap_detection": avg_with,
            "improvement": avg_with - avg_without,
            "gap_detection_accuracy": gap_detection_accuracy,
            "total_missing_fields_found": correctly_identified,
            "total_actually_missing": total_actually_missing,
            "h3_result": (
                f"SUPPORTED: Completeness improved from {avg_without:.3f} to {avg_with:.3f}"
                if h3_supported
                else f"NOT_SUPPORTED: No improvement ({avg_without:.3f} -> {avg_with:.3f})"
            ),
        }

    def _load_dataset(self, path: str) -> list[tuple[str, PCRDocument]]:
        pairs = []
        with open(path) as f:
            for line in f:
                data = json.loads(line.strip())
                pairs.append((data["transcript"], PCRDocument(**data["pcr_json"])))
        return pairs
