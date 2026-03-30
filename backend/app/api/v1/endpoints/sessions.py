"""Session management endpoints."""

from fastapi import APIRouter, HTTPException

from app.dependencies import get_session_manager
from app.schemas.session import (
    SessionCreateRequest,
    SessionDetailResponse,
    SessionResponse,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest = SessionCreateRequest()):
    """Create a new PCR session."""
    manager = get_session_manager()
    session = await manager.create_session(incident_id=request.incident_id)
    pcr_state = session.pcr_manager.get_state()
    return SessionResponse(
        session_id=session.session_id,
        incident_id=session.incident_id,
        created_at=session.created_at,
        status=session.status,
        pcr_state=pcr_state,
    )


@router.get("/{session_id}", response_model=SessionDetailResponse)
async def get_session(session_id: str):
    """Get full session details."""
    manager = get_session_manager()
    session = await manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    pcr_state = session.pcr_manager.get_state()
    return SessionDetailResponse(
        session_id=session.session_id,
        incident_id=session.incident_id,
        created_at=session.created_at,
        status=session.status,
        pcr_state=pcr_state,
        transcript_history=session.transcript_history,
        correction_history=session.correction_history,
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    """Archive and remove a session."""
    manager = get_session_manager()
    success = await manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "archived", "session_id": session_id}


@router.post("/{session_id}/finalize")
async def finalize_session(session_id: str):
    """Mark a session as finalized."""
    manager = get_session_manager()
    session = await manager.finalize_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "finalized", "session_id": session_id}
