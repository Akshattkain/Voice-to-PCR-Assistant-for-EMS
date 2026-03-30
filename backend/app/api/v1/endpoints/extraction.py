"""Extraction endpoints — transcript to structured PCR fields."""

from fastapi import APIRouter, HTTPException

from app.dependencies import get_extraction_service, get_session_manager
from app.schemas.extraction import ComparisonResponse, ExtractionRequest, ExtractionResponse

router = APIRouter(prefix="/sessions/{session_id}", tags=["extraction"])


@router.post("/extract", response_model=ExtractionResponse)
async def extract_pcr(session_id: str, request: ExtractionRequest):
    """Run extraction on a transcript using the specified model."""
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    # Get the extraction service for the requested model
    extractor = get_extraction_service(request.model)

    # Run extraction
    result = await extractor.extract(request.transcript)

    # Apply extraction to PCR state
    pcr_state = session.pcr_manager.apply_extraction(
        extracted=result.pcr,
        confidence_map=result.confidence_map,
        model_name=result.model_name,
    )

    return ExtractionResponse(
        extracted_pcr=result.pcr,
        confidence_map=result.confidence_map,
        model_used=result.model_name,
        latency_ms=result.latency_ms,
        pcr_state=pcr_state,
    )


@router.post("/extract/compare", response_model=ComparisonResponse)
async def compare_extraction(session_id: str, request: ExtractionRequest):
    """Run both extractors and return side-by-side comparison."""
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Run both extractors
    finetuned = get_extraction_service("finetuned")
    baseline = get_extraction_service("llm_baseline")

    ft_result = await finetuned.extract(request.transcript)
    bl_result = await baseline.extract(request.transcript)

    # Compute field diffs
    ft_dict = ft_result.pcr.model_dump()
    bl_dict = bl_result.pcr.model_dump()
    field_diffs = {}
    for field_name in ft_dict:
        if ft_dict[field_name] != bl_dict[field_name]:
            field_diffs[field_name] = {
                "finetuned": ft_dict[field_name],
                "llm_baseline": bl_dict[field_name],
            }

    # Apply the default model's result to state
    pcr_state = session.pcr_manager.get_state()

    return ComparisonResponse(
        finetuned_result=ExtractionResponse(
            extracted_pcr=ft_result.pcr,
            confidence_map=ft_result.confidence_map,
            model_used=ft_result.model_name,
            latency_ms=ft_result.latency_ms,
            pcr_state=pcr_state,
        ),
        llm_baseline_result=ExtractionResponse(
            extracted_pcr=bl_result.pcr,
            confidence_map=bl_result.confidence_map,
            model_used=bl_result.model_name,
            latency_ms=bl_result.latency_ms,
            pcr_state=pcr_state,
        ),
        field_diffs=field_diffs,
    )
