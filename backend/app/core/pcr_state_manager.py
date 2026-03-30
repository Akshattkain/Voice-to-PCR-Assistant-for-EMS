"""PCR State Manager — maintains the evolving PCR document for a session.

Handles confidence-gated field merging, correction overrides, list union
deduplication, version tracking, and completeness calculation.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from app.schemas.nemsis import FIELD_REGISTRY, NEMSISUsage, get_mandatory_fields, get_required_fields
from app.schemas.pcr import FieldConfidence, PCRDocument, PCRStateEnvelope


class PCRStateManager:
    """Manages the evolving PCR document for a single session."""

    def __init__(self, session_id: str, confidence_threshold: float = 0.5):
        self.session_id = session_id
        self.confidence_threshold = confidence_threshold
        self._pcr = PCRDocument()
        self._field_confidence: dict[str, FieldConfidence] = {}
        self._version = 0

    def get_state(self) -> PCRStateEnvelope:
        """Return the current PCR state with metadata."""
        missing_mandatory, missing_required = self.get_missing_fields()
        return PCRStateEnvelope(
            session_id=self.session_id,
            pcr=self._pcr.model_copy(),
            field_confidence=dict(self._field_confidence),
            missing_mandatory=missing_mandatory,
            missing_required=missing_required,
            completeness_score=self.compute_completeness(),
            last_updated=datetime.now(timezone.utc),
            version=self._version,
        )

    def apply_extraction(
        self,
        extracted: PCRDocument,
        confidence_map: dict[str, float],
        model_name: str,
    ) -> PCRStateEnvelope:
        """Merge extracted fields into current state with confidence gating.

        Rules:
        - New fields (currently None/empty): accept if confidence >= threshold
        - Existing scalar fields: overwrite only if new confidence > existing
        - List fields: union merge with deduplication
        """
        extracted_dict = extracted.model_dump()
        now = datetime.now(timezone.utc)

        for field_name, new_value in extracted_dict.items():
            if field_name not in FIELD_REGISTRY:
                continue

            new_confidence = confidence_map.get(field_name, 0.5)

            # Skip null/empty new values
            if self._is_empty(new_value):
                continue

            # Skip below confidence threshold
            if new_confidence < self.confidence_threshold:
                continue

            current_value = getattr(self._pcr, field_name)
            meta = FIELD_REGISTRY[field_name]

            if meta.value_type.startswith("list"):
                # List fields: union merge
                self._merge_list_field(field_name, current_value, new_value)
            elif self._is_empty(current_value):
                # New field: accept
                setattr(self._pcr, field_name, new_value)
            elif field_name in self._field_confidence:
                # Existing field: overwrite only if higher confidence
                existing_conf = self._field_confidence[field_name].confidence
                if new_confidence > existing_conf:
                    setattr(self._pcr, field_name, new_value)
            else:
                # No prior confidence record, accept new value
                setattr(self._pcr, field_name, new_value)

            # Update confidence tracking
            self._field_confidence[field_name] = FieldConfidence(
                value=getattr(self._pcr, field_name),
                confidence=new_confidence,
                source="asr_extraction",
                timestamp=now,
                extraction_model=model_name,
            )

        self._version += 1
        return self.get_state()

    def apply_correction(
        self,
        field_name: str,
        new_value: Any,
        action: str = "update",
    ) -> PCRStateEnvelope:
        """Apply a user correction. Always overrides with confidence=1.0."""
        now = datetime.now(timezone.utc)

        if action == "update":
            setattr(self._pcr, field_name, new_value)
        elif action == "append":
            current = getattr(self._pcr, field_name)
            if isinstance(current, list):
                current.append(new_value)
                setattr(self._pcr, field_name, current)
        elif action == "remove":
            current = getattr(self._pcr, field_name)
            if isinstance(current, list):
                updated = [item for item in current if item != new_value]
                setattr(self._pcr, field_name, updated)
        elif action == "clear":
            meta = FIELD_REGISTRY.get(field_name)
            if meta and meta.value_type.startswith("list"):
                setattr(self._pcr, field_name, [])
            else:
                setattr(self._pcr, field_name, None)

        self._field_confidence[field_name] = FieldConfidence(
            value=getattr(self._pcr, field_name),
            confidence=1.0,
            source="user_correction",
            timestamp=now,
            extraction_model="manual",
        )

        self._version += 1
        return self.get_state()

    def compute_completeness(self) -> float:
        """Ratio of populated mandatory+required fields to total mandatory+required."""
        mandatory = get_mandatory_fields()
        required = get_required_fields()
        all_tracked = mandatory + required

        if not all_tracked:
            return 1.0

        filled = sum(1 for f in all_tracked if not self._is_empty(getattr(self._pcr, f, None)))
        return filled / len(all_tracked)

    def get_missing_fields(self) -> tuple[list[str], list[str]]:
        """Returns (missing_mandatory, missing_required) field names."""
        missing_mandatory = [
            f for f in get_mandatory_fields()
            if self._is_empty(getattr(self._pcr, f, None))
        ]
        missing_required = [
            f for f in get_required_fields()
            if self._is_empty(getattr(self._pcr, f, None))
        ]
        return missing_mandatory, missing_required

    def export_pcr(self) -> PCRDocument:
        """Return a copy of the current PCR document."""
        return self._pcr.model_copy()

    def _merge_list_field(
        self, field_name: str, current: list, new_items: list
    ) -> None:
        """Union merge lists with basic deduplication."""
        if not isinstance(current, list):
            current = []
        if not isinstance(new_items, list):
            return

        existing_normalized = {self._normalize(item) for item in current}
        for item in new_items:
            if self._normalize(item) not in existing_normalized:
                current.append(item)
                existing_normalized.add(self._normalize(item))

        setattr(self._pcr, field_name, current)

    @staticmethod
    def _normalize(value: Any) -> str:
        """Normalize a value for deduplication comparison."""
        if isinstance(value, str):
            return value.lower().strip()
        if isinstance(value, dict):
            # For MedicationGiven-like dicts, normalize by drug name
            return str(value.get("drug", "")).lower().strip()
        return str(value).lower().strip()

    @staticmethod
    def _is_empty(value: Any) -> bool:
        """Check if a value is empty/null."""
        if value is None:
            return True
        if isinstance(value, (list, str)) and len(value) == 0:
            return True
        return False
