"""Session request/response models."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from app.schemas.pcr import PCRStateEnvelope


class SessionCreateRequest(BaseModel):
    incident_id: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str
    incident_id: Optional[str] = None
    created_at: datetime
    status: str = "active"
    pcr_state: PCRStateEnvelope


class TranscriptSegment(BaseModel):
    """A single transcript segment with timestamp."""

    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    timestamp: datetime


class CorrectionEvent(BaseModel):
    """Record of a correction applied to the PCR."""

    utterance: str
    field: str
    old_value: Optional[str] = None
    new_value: str
    timestamp: datetime


class SessionDetailResponse(BaseModel):
    session_id: str
    incident_id: Optional[str] = None
    created_at: datetime
    status: str
    pcr_state: PCRStateEnvelope
    transcript_history: list[TranscriptSegment] = Field(default_factory=list)
    correction_history: list[CorrectionEvent] = Field(default_factory=list)
