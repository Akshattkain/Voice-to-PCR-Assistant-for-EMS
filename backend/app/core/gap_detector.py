"""Gap Detector — checks PCR state against NEMSIS mandatory/required fields.

Generates natural language prompts for missing fields with priority ordering.
Uses template-based prompts (deterministic, no LLM call needed).
"""

from pydantic import BaseModel, Field

from app.schemas.nemsis import FIELD_REGISTRY, NEMSISUsage, FieldMetadata
from app.schemas.pcr import PCRStateEnvelope


class GapItem(BaseModel):
    """A single missing field with metadata."""

    field_name: str
    usage: NEMSISUsage
    section: str
    description: str
    prompt: str
    priority: int  # Lower = higher priority


class GapDetectionResult(BaseModel):
    """Result of gap detection."""

    missing_mandatory: list[GapItem] = Field(default_factory=list)
    missing_required: list[GapItem] = Field(default_factory=list)
    missing_recommended: list[GapItem] = Field(default_factory=list)
    suggested_prompts: list[str] = Field(default_factory=list)
    total_gaps: int = 0


# Priority ordering within each usage level (lower = higher priority)
FIELD_PRIORITY: dict[str, int] = {
    # Mandatory fields (highest priority)
    "age": 1,
    "sex": 2,
    "chief_complaint": 3,
    "primary_impression": 4,
    # Required vitals (high priority)
    "bp_systolic": 10,
    "bp_diastolic": 11,
    "heart_rate": 12,
    "respiratory_rate": 13,
    "spo2": 14,
    "gcs_total": 15,
    "avpu": 16,
    # Required history (medium priority)
    "allergies": 20,
    "past_medical_history": 21,
    "medications_given": 22,
    "procedures": 23,
    "narrative_text": 24,
    # Recommended (lower priority)
    "pain_scale": 30,
    "signs_symptoms": 31,
    "events_leading": 32,
    "medications_current": 33,
    "secondary_impression": 34,
    "temperature": 35,
    "blood_glucose": 36,
    "etco2": 37,
    "cardiac_rhythm": 38,
}


class GapDetector:
    """Checks PCR state against NEMSIS mandatory/required fields."""

    def __init__(self):
        self.field_registry = FIELD_REGISTRY

    def detect_gaps(self, pcr_state: PCRStateEnvelope) -> GapDetectionResult:
        """Identify all missing fields by usage level."""
        pcr_dict = pcr_state.pcr.model_dump()
        missing_mandatory = []
        missing_required = []
        missing_recommended = []

        for field_name, meta in self.field_registry.items():
            value = pcr_dict.get(field_name)
            if self._is_empty(value):
                gap = GapItem(
                    field_name=field_name,
                    usage=meta.usage,
                    section=meta.section.value,
                    description=meta.description,
                    prompt=meta.prompt_template,
                    priority=FIELD_PRIORITY.get(field_name, 50),
                )
                if meta.usage == NEMSISUsage.MANDATORY:
                    missing_mandatory.append(gap)
                elif meta.usage == NEMSISUsage.REQUIRED:
                    missing_required.append(gap)
                else:
                    missing_recommended.append(gap)

        # Sort each list by priority
        missing_mandatory.sort(key=lambda g: g.priority)
        missing_required.sort(key=lambda g: g.priority)
        missing_recommended.sort(key=lambda g: g.priority)

        # Generate prompts (mandatory + required only)
        suggested_prompts = [g.prompt for g in missing_mandatory + missing_required]

        total = len(missing_mandatory) + len(missing_required) + len(missing_recommended)

        return GapDetectionResult(
            missing_mandatory=missing_mandatory,
            missing_required=missing_required,
            missing_recommended=missing_recommended,
            suggested_prompts=suggested_prompts,
            total_gaps=total,
        )

    def generate_batch_prompt(self, gaps: GapDetectionResult) -> str:
        """Generate a single batched prompt for all missing mandatory/required fields."""
        if not gaps.missing_mandatory and not gaps.missing_required:
            return "All mandatory and required fields are complete."

        parts = []
        if gaps.missing_mandatory:
            fields = ", ".join(g.field_name for g in gaps.missing_mandatory)
            parts.append(f"Missing mandatory fields: {fields}")
        if gaps.missing_required:
            fields = ", ".join(g.field_name for g in gaps.missing_required)
            parts.append(f"Missing required fields: {fields}")

        return ". ".join(parts)

    @staticmethod
    def _is_empty(value) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, str)) and len(value) == 0:
            return True
        return False
