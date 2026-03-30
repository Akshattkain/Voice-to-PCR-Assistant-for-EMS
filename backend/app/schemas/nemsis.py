"""NEMSIS field registry — the foundational data structure for the entire system.

Defines all PCR fields extractable from paramedic speech, their NEMSIS element IDs,
usage levels (mandatory/required/recommended), sections, and value types.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class NEMSISUsage(str, Enum):
    """NEMSIS field usage classification."""

    MANDATORY = "mandatory"  # Must be completed, no NOT values allowed
    REQUIRED = "required"  # Must be completed, NOT values allowed
    RECOMMENDED = "recommended"  # Should be completed when applicable


class NEMSISSection(str, Enum):
    """NEMSIS data sections."""

    E_PATIENT = "ePatient"
    E_SITUATION = "eSituation"
    E_HISTORY = "eHistory"
    E_VITALS = "eVitals"
    E_MEDICATIONS = "eMedications"
    E_PROCEDURES = "eProcedures"
    E_NARRATIVE = "eNarrative"


class FieldMetadata(BaseModel):
    """Metadata for a single PCR field mapped to NEMSIS."""

    nemsis_element: str  # e.g., "ePatient.15"
    usage: NEMSISUsage
    section: NEMSISSection
    description: str
    value_type: str  # "int", "str", "float", "list[str]", "list[MedicationGiven]"
    allowed_values: Optional[list[str]] = None
    prompt_template: str  # Used by gap detector to ask for missing values


# Central field registry: maps PCR JSON field names to NEMSIS metadata
FIELD_REGISTRY: dict[str, FieldMetadata] = {
    # --- ePatient section ---
    "age": FieldMetadata(
        nemsis_element="ePatient.15",
        usage=NEMSISUsage.MANDATORY,
        section=NEMSISSection.E_PATIENT,
        description="Patient age in years",
        value_type="int",
        prompt_template="What is the patient's age?",
    ),
    "sex": FieldMetadata(
        nemsis_element="ePatient.13",
        usage=NEMSISUsage.MANDATORY,
        section=NEMSISSection.E_PATIENT,
        description="Patient gender",
        value_type="str",
        allowed_values=["male", "female", "unknown"],
        prompt_template="What is the patient's sex?",
    ),
    # --- eSituation section ---
    "chief_complaint": FieldMetadata(
        nemsis_element="eSituation.04",
        usage=NEMSISUsage.MANDATORY,
        section=NEMSISSection.E_SITUATION,
        description="Primary reason for the EMS encounter",
        value_type="str",
        prompt_template="What is the patient's chief complaint?",
    ),
    "primary_impression": FieldMetadata(
        nemsis_element="eSituation.11",
        usage=NEMSISUsage.MANDATORY,
        section=NEMSISSection.E_SITUATION,
        description="Provider's primary clinical impression",
        value_type="str",
        prompt_template="What is the primary impression or working diagnosis?",
    ),
    "secondary_impression": FieldMetadata(
        nemsis_element="eSituation.12",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_SITUATION,
        description="Provider's secondary clinical impression",
        value_type="str",
        prompt_template="Is there a secondary impression?",
    ),
    "signs_symptoms": FieldMetadata(
        nemsis_element="eSituation.09",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_SITUATION,
        description="Signs and symptoms observed or reported",
        value_type="list[str]",
        prompt_template="What signs and symptoms were observed?",
    ),
    "events_leading": FieldMetadata(
        nemsis_element="eSituation.06",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_SITUATION,
        description="Description of events leading to the incident",
        value_type="str",
        prompt_template="What events led to this incident?",
    ),
    # --- eHistory section ---
    "allergies": FieldMetadata(
        nemsis_element="eHistory.06",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_HISTORY,
        description="Known patient allergies",
        value_type="list[str]",
        prompt_template="Does the patient have any known allergies?",
    ),
    "medications_current": FieldMetadata(
        nemsis_element="eHistory.08",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_HISTORY,
        description="Current medications the patient takes",
        value_type="list[str]",
        prompt_template="What medications is the patient currently taking?",
    ),
    "past_medical_history": FieldMetadata(
        nemsis_element="eHistory.08",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_HISTORY,
        description="Relevant past medical history",
        value_type="list[str]",
        prompt_template="What is the patient's past medical history?",
    ),
    # --- eVitals section ---
    "bp_systolic": FieldMetadata(
        nemsis_element="eVitals.06",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_VITALS,
        description="Systolic blood pressure in mmHg",
        value_type="int",
        prompt_template="What is the systolic blood pressure?",
    ),
    "bp_diastolic": FieldMetadata(
        nemsis_element="eVitals.07",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_VITALS,
        description="Diastolic blood pressure in mmHg",
        value_type="int",
        prompt_template="What is the diastolic blood pressure?",
    ),
    "heart_rate": FieldMetadata(
        nemsis_element="eVitals.10",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_VITALS,
        description="Heart rate in beats per minute",
        value_type="int",
        prompt_template="What is the heart rate?",
    ),
    "respiratory_rate": FieldMetadata(
        nemsis_element="eVitals.14",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_VITALS,
        description="Respiratory rate in breaths per minute",
        value_type="int",
        prompt_template="What is the respiratory rate?",
    ),
    "spo2": FieldMetadata(
        nemsis_element="eVitals.12",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_VITALS,
        description="Pulse oximetry SpO2 percentage",
        value_type="int",
        prompt_template="What is the SpO2?",
    ),
    "gcs_total": FieldMetadata(
        nemsis_element="eVitals.23",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_VITALS,
        description="Glasgow Coma Scale total score",
        value_type="int",
        prompt_template="What is the GCS score?",
    ),
    "avpu": FieldMetadata(
        nemsis_element="eVitals.26",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_VITALS,
        description="AVPU responsiveness level",
        value_type="str",
        allowed_values=["Alert", "Verbal", "Pain", "Unresponsive"],
        prompt_template="What is the patient's AVPU level?",
    ),
    "pain_scale": FieldMetadata(
        nemsis_element="eVitals.27",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_VITALS,
        description="Pain scale rating (0-10)",
        value_type="int",
        prompt_template="What is the patient's pain level on a scale of 0 to 10?",
    ),
    "temperature": FieldMetadata(
        nemsis_element="eVitals.24",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_VITALS,
        description="Body temperature in degrees Fahrenheit",
        value_type="float",
        prompt_template="What is the patient's temperature?",
    ),
    "blood_glucose": FieldMetadata(
        nemsis_element="eVitals.18",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_VITALS,
        description="Blood glucose level in mg/dL",
        value_type="int",
        prompt_template="What is the blood glucose level?",
    ),
    "etco2": FieldMetadata(
        nemsis_element="eVitals.16",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_VITALS,
        description="End-tidal CO2 in mmHg",
        value_type="int",
        prompt_template="What is the EtCO2?",
    ),
    "cardiac_rhythm": FieldMetadata(
        nemsis_element="eVitals.03",
        usage=NEMSISUsage.RECOMMENDED,
        section=NEMSISSection.E_VITALS,
        description="Cardiac rhythm interpretation",
        value_type="str",
        prompt_template="What is the cardiac rhythm?",
    ),
    # --- eMedications section ---
    "medications_given": FieldMetadata(
        nemsis_element="eMedications.03",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_MEDICATIONS,
        description="Medications administered during encounter",
        value_type="list[MedicationGiven]",
        prompt_template="Were any medications administered?",
    ),
    # --- eProcedures section ---
    "procedures": FieldMetadata(
        nemsis_element="eProcedures.03",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_PROCEDURES,
        description="Procedures performed during encounter",
        value_type="list[str]",
        prompt_template="What procedures were performed?",
    ),
    # --- eNarrative section ---
    "narrative_text": FieldMetadata(
        nemsis_element="eNarrative.01",
        usage=NEMSISUsage.REQUIRED,
        section=NEMSISSection.E_NARRATIVE,
        description="Free-text clinical narrative",
        value_type="str",
        prompt_template="Can you provide a brief narrative of the encounter?",
    ),
}


def get_mandatory_fields() -> list[str]:
    """Return field names classified as mandatory."""
    return [name for name, meta in FIELD_REGISTRY.items() if meta.usage == NEMSISUsage.MANDATORY]


def get_required_fields() -> list[str]:
    """Return field names classified as required."""
    return [name for name, meta in FIELD_REGISTRY.items() if meta.usage == NEMSISUsage.REQUIRED]


def get_recommended_fields() -> list[str]:
    """Return field names classified as recommended."""
    return [
        name for name, meta in FIELD_REGISTRY.items() if meta.usage == NEMSISUsage.RECOMMENDED
    ]


def get_fields_by_section(section: NEMSISSection) -> dict[str, FieldMetadata]:
    """Return all fields belonging to a NEMSIS section."""
    return {name: meta for name, meta in FIELD_REGISTRY.items() if meta.section == section}
