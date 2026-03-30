"""Correction request/response models."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.pcr import PCRStateEnvelope


class CorrectionRequest(BaseModel):
    utterance: str  # e.g., "Change heart rate to 108"


class CorrectionIntent(BaseModel):
    """Parsed intent from a natural language correction."""

    field: str
    action: Literal["update", "append", "remove", "clear"]
    value: Any
    confidence: float = Field(ge=0.0, le=1.0)


class CorrectionResponse(BaseModel):
    """Result of applying corrections."""

    applied_corrections: list[CorrectionIntent]
    pcr_state: PCRStateEnvelope
    rejected: list[dict] = Field(default_factory=list)
