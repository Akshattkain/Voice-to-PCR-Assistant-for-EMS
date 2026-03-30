"""Correction endpoint — natural language corrections to PCR fields."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.dependencies import get_openai_client, get_session_manager
from app.schemas.correction import CorrectionRequest, CorrectionResponse
from app.schemas.session import CorrectionEvent
from app.services.correction.correction_handler import CorrectionHandler
from app.services.correction.correction_parser import CorrectionParser

router = APIRouter(prefix="/sessions/{session_id}", tags=["correction"])


@router.post("/correct", response_model=CorrectionResponse)
async def apply_correction(session_id: str, request: CorrectionRequest):
    """Parse and apply a natural language correction to the PCR."""
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    # Parse the correction utterance
    parser = CorrectionParser(openai_client=get_openai_client())
    intents = await parser.parse(
        utterance=request.utterance,
        current_pcr=session.pcr_manager.export_pcr(),
    )

    if not intents:
        raise HTTPException(
            status_code=422,
            detail="Could not parse any corrections from the utterance",
        )

    # Apply corrections
    handler = CorrectionHandler()
    pcr_state, rejected = handler.apply(session.pcr_manager, intents)

    # Log correction events
    for intent in intents:
        if not any(r["intent"]["field"] == intent.field for r in rejected):
            await session_mgr.add_correction(
                session_id,
                CorrectionEvent(
                    utterance=request.utterance,
                    field=intent.field,
                    new_value=str(intent.value),
                    timestamp=datetime.now(timezone.utc),
                ),
            )

    return CorrectionResponse(
        applied_corrections=intents,
        pcr_state=pcr_state,
        rejected=rejected,
    )
