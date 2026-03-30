"""PCR document models with confidence tracking."""

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class MedicationGiven(BaseModel):
    """A medication administered during the encounter."""

    drug: str
    dose: Optional[float] = None
    unit: Optional[str] = None
    route: Optional[str] = None
    time: Optional[str] = None


class PCRDocument(BaseModel):
    """Full PCR with all extractable fields aligned to NEMSIS."""

    # Demographics (ePatient)
    age: Optional[int] = None
    sex: Optional[str] = None

    # Situation (eSituation)
    chief_complaint: Optional[str] = None
    primary_impression: Optional[str] = None
    secondary_impression: Optional[str] = None
    signs_symptoms: list[str] = Field(default_factory=list)
    events_leading: Optional[str] = None

    # History (eHistory)
    allergies: list[str] = Field(default_factory=list)
    medications_current: list[str] = Field(default_factory=list)
    past_medical_history: list[str] = Field(default_factory=list)

    # Vitals (eVitals)
    bp_systolic: Optional[int] = None
    bp_diastolic: Optional[int] = None
    heart_rate: Optional[int] = None
    respiratory_rate: Optional[int] = None
    spo2: Optional[int] = None
    gcs_total: Optional[int] = None
    avpu: Optional[str] = None
    pain_scale: Optional[int] = None
    temperature: Optional[float] = None
    blood_glucose: Optional[int] = None
    etco2: Optional[int] = None
    cardiac_rhythm: Optional[str] = None

    # Medications (eMedications)
    medications_given: list[MedicationGiven] = Field(default_factory=list)

    # Procedures (eProcedures)
    procedures: list[str] = Field(default_factory=list)

    # Narrative (eNarrative)
    narrative_text: Optional[str] = None


class FieldConfidence(BaseModel):
    """Confidence metadata for a single PCR field."""

    value: Any
    confidence: float = Field(ge=0.0, le=1.0)
    source: str  # "asr_extraction", "user_correction", "gap_fill"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extraction_model: str  # "finetuned_t5", "llm_baseline", "manual"


class PCRStateEnvelope(BaseModel):
    """PCR document wrapped with per-field confidence and completeness metadata."""

    session_id: str
    pcr: PCRDocument
    field_confidence: dict[str, FieldConfidence] = Field(default_factory=dict)
    missing_mandatory: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    completeness_score: float = 0.0
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 0
