"""Evaluation metrics — F1, hallucination rate, completeness.

Directly supports testing H1, H2, and H3.
"""

import re
from typing import Any, Optional

from app.schemas.evaluation import (
    CompletenessResult,
    FieldScore,
    HallucinationResult,
)
from app.schemas.nemsis import get_mandatory_fields, get_required_fields
from app.schemas.pcr import PCRDocument


def compute_field_f1(
    predicted: PCRDocument, ground_truth: PCRDocument
) -> dict[str, FieldScore]:
    """Compute per-field precision, recall, and F1.

    For scalar fields: exact match (1.0 or 0.0)
    For string fields: token-level F1 (bag of words overlap)
    For list fields: set-level precision/recall/F1
    """
    pred_dict = predicted.model_dump()
    gt_dict = ground_truth.model_dump()
    scores = {}

    for field_name in pred_dict:
        pred_val = pred_dict[field_name]
        gt_val = gt_dict[field_name]

        if _is_empty(pred_val) and _is_empty(gt_val):
            # Both empty — true negative, skip
            continue

        if isinstance(gt_val, list) or isinstance(pred_val, list):
            scores[field_name] = _list_f1(pred_val or [], gt_val or [])
        elif isinstance(gt_val, (int, float)) or isinstance(pred_val, (int, float)):
            scores[field_name] = _numeric_f1(pred_val, gt_val)
        elif isinstance(gt_val, str) or isinstance(pred_val, str):
            scores[field_name] = _string_f1(pred_val, gt_val)
        else:
            scores[field_name] = _exact_f1(pred_val, gt_val)

    return scores


def compute_aggregate_f1(field_scores: dict[str, FieldScore]) -> dict[str, float]:
    """Compute micro and macro F1 across all fields."""
    if not field_scores:
        return {"micro_f1": 0.0, "macro_f1": 0.0}

    # Macro F1: average of per-field F1
    f1s = [s.f1 for s in field_scores.values()]
    macro_f1 = sum(f1s) / len(f1s)

    # Micro F1: weighted by field presence
    total_p = sum(s.precision for s in field_scores.values())
    total_r = sum(s.recall for s in field_scores.values())
    n = len(field_scores)
    avg_p = total_p / n if n > 0 else 0
    avg_r = total_r / n if n > 0 else 0
    micro_f1 = 2 * avg_p * avg_r / (avg_p + avg_r) if (avg_p + avg_r) > 0 else 0

    return {"micro_f1": micro_f1, "macro_f1": macro_f1}


def compute_hallucination_rate(
    predicted: PCRDocument,
    ground_truth: PCRDocument,
    transcript: str,
) -> HallucinationResult:
    """Compute hallucination rate.

    A field value is a hallucination if:
    1. It is present in prediction but absent/different from ground truth
    2. It is NOT derivable from the transcript text
    """
    pred_dict = predicted.model_dump()
    gt_dict = ground_truth.model_dump()
    transcript_lower = transcript.lower()

    hallucinated = []
    total_predicted = 0

    for field_name, pred_val in pred_dict.items():
        if _is_empty(pred_val):
            continue

        total_predicted += 1
        gt_val = gt_dict.get(field_name)

        # Check if predicted matches ground truth
        if _values_match(pred_val, gt_val):
            continue

        # Predicted differs from ground truth — check if it's in the transcript
        if not _value_in_transcript(pred_val, transcript_lower):
            hallucinated.append(field_name)

    rate = len(hallucinated) / total_predicted if total_predicted > 0 else 0.0

    return HallucinationResult(
        hallucination_count=len(hallucinated),
        total_predicted_fields=total_predicted,
        hallucination_rate=rate,
        hallucinated_fields=hallucinated,
    )


def compute_completeness(predicted: PCRDocument) -> CompletenessResult:
    """Compute field completeness against NEMSIS mandatory/required fields."""
    pred_dict = predicted.model_dump()

    mandatory = get_mandatory_fields()
    required = get_required_fields()

    mandatory_filled = sum(1 for f in mandatory if not _is_empty(pred_dict.get(f)))
    required_filled = sum(1 for f in required if not _is_empty(pred_dict.get(f)))

    mandatory_total = len(mandatory)
    required_total = len(required)
    total = mandatory_total + required_total
    filled = mandatory_filled + required_filled

    return CompletenessResult(
        mandatory_filled=mandatory_filled,
        mandatory_total=mandatory_total,
        mandatory_rate=mandatory_filled / mandatory_total if mandatory_total > 0 else 1.0,
        required_filled=required_filled,
        required_total=required_total,
        required_rate=required_filled / required_total if required_total > 0 else 1.0,
        overall_completeness=filled / total if total > 0 else 1.0,
        missing_mandatory=[f for f in mandatory if _is_empty(pred_dict.get(f))],
        missing_required=[f for f in required if _is_empty(pred_dict.get(f))],
    )


# --- Helper functions ---

def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, str)) and len(value) == 0:
        return True
    return False


def _normalize_str(s: str) -> str:
    """Normalize a string for comparison."""
    return re.sub(r"[^a-z0-9\s]", "", s.lower()).strip()


def _tokenize(s: str) -> set[str]:
    """Tokenize a string into a set of words."""
    return set(_normalize_str(s).split())


def _numeric_f1(pred: Any, gt: Any, tolerance: int = 2) -> FieldScore:
    """F1 for numeric fields with tolerance."""
    if pred is None or gt is None:
        if pred is None and gt is None:
            return FieldScore(precision=1.0, recall=1.0, f1=1.0)
        return FieldScore(precision=0.0, recall=0.0, f1=0.0)

    try:
        match = abs(float(pred) - float(gt)) <= tolerance
    except (TypeError, ValueError):
        match = False

    score = 1.0 if match else 0.0
    return FieldScore(precision=score, recall=score, f1=score)


def _string_f1(pred: Any, gt: Any) -> FieldScore:
    """Token-level F1 for string fields."""
    if pred is None and gt is None:
        return FieldScore(precision=1.0, recall=1.0, f1=1.0)
    if pred is None or gt is None:
        return FieldScore(precision=0.0, recall=0.0, f1=0.0)

    pred_tokens = _tokenize(str(pred))
    gt_tokens = _tokenize(str(gt))

    if not pred_tokens and not gt_tokens:
        return FieldScore(precision=1.0, recall=1.0, f1=1.0)
    if not pred_tokens or not gt_tokens:
        return FieldScore(precision=0.0, recall=0.0, f1=0.0)

    overlap = pred_tokens & gt_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return FieldScore(precision=precision, recall=recall, f1=f1)


def _list_f1(pred: list, gt: list) -> FieldScore:
    """Set-level F1 for list fields."""
    if not pred and not gt:
        return FieldScore(precision=1.0, recall=1.0, f1=1.0)
    if not pred or not gt:
        return FieldScore(precision=0.0, recall=0.0, f1=0.0)

    # Normalize items for comparison
    pred_normalized = {_normalize_item(item) for item in pred}
    gt_normalized = {_normalize_item(item) for item in gt}

    overlap = pred_normalized & gt_normalized
    precision = len(overlap) / len(pred_normalized)
    recall = len(overlap) / len(gt_normalized)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return FieldScore(precision=precision, recall=recall, f1=f1)


def _normalize_item(item: Any) -> str:
    """Normalize a list item for comparison."""
    if isinstance(item, dict):
        # For MedicationGiven-like dicts, normalize by drug name
        return _normalize_str(item.get("drug", str(item)))
    return _normalize_str(str(item))


def _exact_f1(pred: Any, gt: Any) -> FieldScore:
    """Exact match F1."""
    match = pred == gt
    score = 1.0 if match else 0.0
    return FieldScore(precision=score, recall=score, f1=score)


def _values_match(pred: Any, gt: Any) -> bool:
    """Check if predicted value matches ground truth."""
    if _is_empty(pred) and _is_empty(gt):
        return True
    if _is_empty(pred) or _is_empty(gt):
        return False

    if isinstance(pred, (int, float)) and isinstance(gt, (int, float)):
        return abs(float(pred) - float(gt)) <= 2

    if isinstance(pred, str) and isinstance(gt, str):
        return _normalize_str(pred) == _normalize_str(gt)

    if isinstance(pred, list) and isinstance(gt, list):
        pred_set = {_normalize_item(i) for i in pred}
        gt_set = {_normalize_item(i) for i in gt}
        return pred_set == gt_set

    return pred == gt


def _value_in_transcript(value: Any, transcript_lower: str) -> bool:
    """Check if a value is derivable from the transcript text."""
    if isinstance(value, (int, float)):
        return str(int(value)) in transcript_lower

    if isinstance(value, str):
        # Check for substantial overlap
        tokens = _tokenize(value)
        if not tokens:
            return False
        matches = sum(1 for t in tokens if t in transcript_lower)
        return matches / len(tokens) >= 0.5

    if isinstance(value, list):
        return any(_value_in_transcript(item, transcript_lower) for item in value)

    if isinstance(value, dict):
        # For medications: check drug name
        drug = value.get("drug", "")
        return drug.lower() in transcript_lower if drug else False

    return False
