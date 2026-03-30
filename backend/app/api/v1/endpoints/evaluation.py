"""Evaluation endpoint — run evaluation via API."""

from fastapi import APIRouter, HTTPException

from app.dependencies import get_extraction_service
from app.schemas.evaluation import BatchComparisonReport, BatchEvalRequest, SingleEvalResult
from app.schemas.pcr import PCRDocument
from evaluation.comparator import ModelComparator

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.post("/single", response_model=SingleEvalResult)
async def evaluate_single(
    transcript: str,
    ground_truth: dict,
    model: str = "llm_baseline",
):
    """Evaluate a single (transcript, ground_truth) pair."""
    gt_pcr = PCRDocument(**ground_truth)
    extractor = get_extraction_service(model)

    comparator = ModelComparator()
    result = await comparator.evaluate_single(transcript, gt_pcr, extractor)
    return result


@router.post("/batch", response_model=BatchComparisonReport)
async def evaluate_batch(request: BatchEvalRequest):
    """Run batch evaluation on a dataset."""
    finetuned = None
    baseline = None

    try:
        if "finetuned" in request.models:
            finetuned = get_extraction_service("finetuned")
        if "llm_baseline" in request.models:
            baseline = get_extraction_service("llm_baseline")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    comparator = ModelComparator(
        finetuned_service=finetuned,
        baseline_service=baseline,
    )

    try:
        report = await comparator.compare_batch(request.dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset file not found")

    return report
