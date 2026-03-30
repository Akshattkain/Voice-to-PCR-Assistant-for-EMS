"""ASR transcription request/response models."""

from typing import Optional

from pydantic import BaseModel


class TranscriptionSegment(BaseModel):
    """A timestamped segment from ASR output."""

    text: str
    start: float  # seconds
    end: float  # seconds


class TranscriptionResponse(BaseModel):
    """Response from the ASR transcription service."""

    transcript_text: str
    segments: list[TranscriptionSegment] = []
    language: str = "en"
    duration_sec: Optional[float] = None
    model_used: str = "whisper-1"
