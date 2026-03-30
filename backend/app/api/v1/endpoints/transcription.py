"""Transcription endpoint — upload audio and get transcript."""

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, UploadFile

from app.dependencies import get_asr_service, get_session_manager
from app.schemas.session import TranscriptSegment
from app.schemas.transcription import TranscriptionResponse
from app.utils.audio import validate_audio_format, validate_audio_size

router = APIRouter(prefix="/sessions/{session_id}", tags=["transcription"])


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(session_id: str, file: UploadFile):
    """Upload an audio file and transcribe it via Whisper."""
    # Validate session exists
    session_mgr = get_session_manager()
    session = await session_mgr.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if session.status != "active":
        raise HTTPException(status_code=400, detail="Session is not active")

    # Validate audio
    audio_format = validate_audio_format(file.filename or "audio.wav")
    audio_data = await file.read()
    validate_audio_size(audio_data)

    # Transcribe
    asr = get_asr_service()
    result = await asr.transcribe(audio_data, audio_format=audio_format)

    # Store transcript in session history
    await session_mgr.add_transcript(
        session_id,
        TranscriptSegment(
            text=result.transcript_text,
            start_time=0.0,
            end_time=result.duration_sec,
            timestamp=datetime.now(timezone.utc),
        ),
    )

    return result
