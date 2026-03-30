"""Session Manager — in-memory session store.

Manages active PCR sessions with their state managers, transcript history,
and correction logs. Designed for MVP with interface suitable for Redis swap.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.core.pcr_state_manager import PCRStateManager
from app.schemas.pcr import PCRStateEnvelope
from app.schemas.session import CorrectionEvent, TranscriptSegment


class SessionData(BaseModel):
    """Data associated with a single PCR session."""

    model_config = {"arbitrary_types_allowed": True}

    session_id: str
    incident_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "active"  # "active", "finalized", "archived"
    transcript_history: list[TranscriptSegment] = Field(default_factory=list)
    correction_history: list[CorrectionEvent] = Field(default_factory=list)
    # PCR state manager is not serialized — managed separately
    _pcr_manager: Optional[PCRStateManager] = None

    @property
    def pcr_manager(self) -> PCRStateManager:
        if self._pcr_manager is None:
            self._pcr_manager = PCRStateManager(session_id=self.session_id)
        return self._pcr_manager

    @pcr_manager.setter
    def pcr_manager(self, value: PCRStateManager):
        self._pcr_manager = value


class SessionManager:
    """Manages active sessions. In-memory dict for MVP."""

    def __init__(self):
        self._sessions: dict[str, SessionData] = {}

    async def create_session(self, incident_id: Optional[str] = None) -> SessionData:
        """Create a new PCR session."""
        session_id = str(uuid.uuid4())
        session = SessionData(
            session_id=session_id,
            incident_id=incident_id,
        )
        session.pcr_manager = PCRStateManager(session_id=session_id)
        self._sessions[session_id] = session
        return session

    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    async def delete_session(self, session_id: str) -> bool:
        """Archive and remove a session."""
        if session_id in self._sessions:
            self._sessions[session_id].status = "archived"
            del self._sessions[session_id]
            return True
        return False

    async def finalize_session(self, session_id: str) -> Optional[SessionData]:
        """Mark a session as finalized (no more updates)."""
        session = self._sessions.get(session_id)
        if session:
            session.status = "finalized"
        return session

    async def add_transcript(
        self, session_id: str, segment: TranscriptSegment
    ) -> Optional[SessionData]:
        """Add a transcript segment to session history."""
        session = self._sessions.get(session_id)
        if session:
            session.transcript_history.append(segment)
        return session

    async def add_correction(
        self, session_id: str, event: CorrectionEvent
    ) -> Optional[SessionData]:
        """Log a correction event."""
        session = self._sessions.get(session_id)
        if session:
            session.correction_history.append(event)
        return session

    async def list_sessions(self) -> list[SessionData]:
        """List all active sessions."""
        return [s for s in self._sessions.values() if s.status == "active"]

    def get_pcr_state(self, session_id: str) -> Optional[PCRStateEnvelope]:
        """Get the current PCR state for a session."""
        session = self._sessions.get(session_id)
        if session:
            return session.pcr_manager.get_state()
        return None
