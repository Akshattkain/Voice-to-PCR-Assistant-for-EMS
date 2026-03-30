"""Gap detection endpoint — check for missing mandatory/required PCR fields."""

from fastapi import APIRouter, HTTPException

from app.core.gap_detector import GapDetectionResult
from app.dependencies import get_gap_detector, get_session_manager

router = APIRouter(prefix="/sessions/{session_id}", tags=["gap_detection"])


@router.get("/gaps", response_model=GapDetectionResult)
async def detect_gaps(session_id: str):
    """Check current PCR state for missing mandatory and required fields."""
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    pcr_state = session.pcr_manager.get_state()
    detector = get_gap_detector()
    return detector.detect_gaps(pcr_state)
