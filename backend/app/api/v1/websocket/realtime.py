"""WebSocket endpoint for real-time PCR session interaction.

Supports: audio streaming, real-time extraction, corrections, gap alerts.
"""

import base64
import json
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.dependencies import (
    get_asr_service,
    get_extraction_service,
    get_gap_detector,
    get_openai_client,
    get_session_manager,
)
from app.schemas.session import TranscriptSegment
from app.schemas.websocket import WSServerMessage
from app.services.correction.correction_handler import CorrectionHandler
from app.services.correction.correction_parser import CorrectionParser
from app.utils.logging import logger

router = APIRouter()


async def send_message(ws: WebSocket, msg_type: str, payload: dict, version: int = 0):
    """Send a typed message to the client."""
    message = WSServerMessage(
        type=msg_type,
        payload=payload,
        timestamp=datetime.now(timezone.utc),
        version=version,
    )
    await ws.send_json(message.model_dump(mode="json"))


@router.websocket("/ws/session/{session_id}")
async def websocket_session(ws: WebSocket, session_id: str):
    """Bidirectional WebSocket for real-time PCR session."""
    await ws.accept()

    session_mgr = get_session_manager()
    session = await session_mgr.get_session(session_id)
    if not session:
        await send_message(ws, "error", {"message": "Session not found"})
        await ws.close()
        return

    # Send initial state
    state = session.pcr_manager.get_state()
    await send_message(ws, "pcr_state", state.model_dump(mode="json"), state.version)

    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                await send_message(ws, "error", {"message": "Invalid JSON"})
                continue

            msg_type = msg.get("type", "")
            payload = msg.get("payload", {})

            if msg_type == "audio_chunk":
                await _handle_audio_chunk(ws, session, payload)
            elif msg_type == "correction":
                await _handle_correction(ws, session, payload)
            elif msg_type == "request_gaps":
                await _handle_gap_request(ws, session)
            elif msg_type == "finalize":
                await _handle_finalize(ws, session)
                break
            else:
                await send_message(ws, "error", {"message": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")


async def _handle_audio_chunk(ws: WebSocket, session, payload: dict):
    """Process an audio chunk: transcribe and extract."""
    audio_b64 = payload.get("audio_base64", "")
    audio_format = payload.get("format", "wav")

    if not audio_b64:
        await send_message(ws, "error", {"message": "No audio data"})
        return

    try:
        audio_data = base64.b64decode(audio_b64)
    except Exception:
        await send_message(ws, "error", {"message": "Invalid base64 audio"})
        return

    # Transcribe
    asr = get_asr_service()
    result = await asr.transcribe(audio_data, audio_format=audio_format)

    # Store transcript
    await get_session_manager().add_transcript(
        session.session_id,
        TranscriptSegment(
            text=result.transcript_text,
            start_time=0.0,
            end_time=result.duration_sec,
            timestamp=datetime.now(timezone.utc),
        ),
    )

    # Send transcript
    await send_message(ws, "transcript_final", {
        "text": result.transcript_text,
        "segments": [s.model_dump() for s in result.segments],
    })

    # Run extraction
    extractor = get_extraction_service()
    extraction = await extractor.extract(result.transcript_text)

    # Apply to state
    state = session.pcr_manager.apply_extraction(
        extracted=extraction.pcr,
        confidence_map=extraction.confidence_map,
        model_name=extraction.model_name,
    )

    await send_message(ws, "extraction_update", {
        "extracted_fields": extraction.pcr.model_dump(exclude_none=True),
        "confidence_map": extraction.confidence_map,
        "model_used": extraction.model_name,
        "latency_ms": extraction.latency_ms,
    }, state.version)

    # Send full state
    await send_message(ws, "pcr_state", state.model_dump(mode="json"), state.version)

    # Auto gap detection
    detector = get_gap_detector()
    gaps = detector.detect_gaps(state)
    if gaps.missing_mandatory:
        await send_message(ws, "gap_alert", {
            "missing_mandatory": [g.model_dump() for g in gaps.missing_mandatory],
            "missing_required": [g.model_dump() for g in gaps.missing_required],
            "suggested_prompts": gaps.suggested_prompts[:3],  # Top 3 prompts
        }, state.version)


async def _handle_correction(ws: WebSocket, session, payload: dict):
    """Process a verbal correction."""
    utterance = payload.get("utterance", "")
    if not utterance:
        await send_message(ws, "error", {"message": "No utterance provided"})
        return

    parser = CorrectionParser(openai_client=get_openai_client())
    intents = await parser.parse(utterance, session.pcr_manager.export_pcr())

    if not intents:
        await send_message(ws, "error", {"message": "Could not parse correction"})
        return

    handler = CorrectionHandler()
    state, rejected = handler.apply(session.pcr_manager, intents)

    await send_message(ws, "correction_applied", {
        "applied": [i.model_dump() for i in intents],
        "rejected": rejected,
    }, state.version)

    await send_message(ws, "pcr_state", state.model_dump(mode="json"), state.version)


async def _handle_gap_request(ws: WebSocket, session):
    """Handle explicit gap detection request."""
    state = session.pcr_manager.get_state()
    detector = get_gap_detector()
    gaps = detector.detect_gaps(state)

    await send_message(ws, "gap_alert", {
        "missing_mandatory": [g.model_dump() for g in gaps.missing_mandatory],
        "missing_required": [g.model_dump() for g in gaps.missing_required],
        "missing_recommended": [g.model_dump() for g in gaps.missing_recommended],
        "suggested_prompts": gaps.suggested_prompts,
        "total_gaps": gaps.total_gaps,
    }, state.version)


async def _handle_finalize(ws: WebSocket, session):
    """Finalize the session and send final state."""
    mgr = get_session_manager()
    await mgr.finalize_session(session.session_id)
    state = session.pcr_manager.get_state()

    await send_message(ws, "pcr_state", {
        **state.model_dump(mode="json"),
        "finalized": True,
    }, state.version)
