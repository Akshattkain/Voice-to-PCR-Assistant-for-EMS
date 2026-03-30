"""PCR export endpoint — export the current PCR as JSON."""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.dependencies import get_session_manager
from app.schemas.pcr import PCRDocument, PCRStateEnvelope

router = APIRouter(prefix="/sessions/{session_id}", tags=["pcr_export"])


@router.get("/pcr", response_model=PCRStateEnvelope)
async def export_pcr(session_id: str):
    """Export the current PCR state with confidence metadata."""
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.pcr_manager.get_state()


@router.get("/pcr/json")
async def export_pcr_json(session_id: str):
    """Export just the PCR document as clean JSON (no metadata)."""
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    pcr = session.pcr_manager.export_pcr()
    return JSONResponse(
        content=pcr.model_dump(exclude_none=True),
        media_type="application/json",
    )
