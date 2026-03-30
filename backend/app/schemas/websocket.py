"""WebSocket message types for real-time communication."""

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class WSClientMessage(BaseModel):
    """Message from client to server."""

    type: Literal["audio_chunk", "correction", "request_gaps", "finalize"]
    payload: dict[str, Any] = Field(default_factory=dict)
    # audio_chunk: {"audio_base64": str, "sample_rate": int, "format": str}
    # correction: {"utterance": str}
    # request_gaps: {}
    # finalize: {}


class WSServerMessage(BaseModel):
    """Message from server to client."""

    type: Literal[
        "transcript_partial",
        "transcript_final",
        "extraction_update",
        "pcr_state",
        "gap_alert",
        "correction_applied",
        "error",
    ]
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 0  # PCR state version for optimistic concurrency
