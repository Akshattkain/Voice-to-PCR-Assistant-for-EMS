"""Tests for Gap Detector."""

from app.core.gap_detector import GapDetector
from app.schemas.pcr import PCRDocument, PCRStateEnvelope


def _make_state(pcr: PCRDocument, session_id: str = "test") -> PCRStateEnvelope:
    return PCRStateEnvelope(session_id=session_id, pcr=pcr)


def test_empty_pcr_has_all_gaps():
    detector = GapDetector()
    state = _make_state(PCRDocument())
    result = detector.detect_gaps(state)

    assert len(result.missing_mandatory) > 0
    assert len(result.missing_required) > 0
    mandatory_names = [g.field_name for g in result.missing_mandatory]
    assert "age" in mandatory_names
    assert "sex" in mandatory_names
    assert "chief_complaint" in mandatory_names


def test_filled_mandatory_not_in_gaps():
    detector = GapDetector()
    pcr = PCRDocument(
        age=71, sex="male", chief_complaint="Chest pain",
        primary_impression="Angina"
    )
    state = _make_state(pcr)
    result = detector.detect_gaps(state)

    mandatory_names = [g.field_name for g in result.missing_mandatory]
    assert "age" not in mandatory_names
    assert "sex" not in mandatory_names
    assert len(result.missing_mandatory) == 0


def test_prompts_generated_for_missing():
    detector = GapDetector()
    state = _make_state(PCRDocument())
    result = detector.detect_gaps(state)

    assert len(result.suggested_prompts) > 0
    # Should include prompts for mandatory and required fields
    assert any("age" in p.lower() for p in result.suggested_prompts)


def test_priority_ordering():
    detector = GapDetector()
    state = _make_state(PCRDocument())
    result = detector.detect_gaps(state)

    # Mandatory gaps should be sorted by priority
    if len(result.missing_mandatory) > 1:
        priorities = [g.priority for g in result.missing_mandatory]
        assert priorities == sorted(priorities)


def test_fully_filled_pcr_has_no_mandatory_gaps():
    detector = GapDetector()
    pcr = PCRDocument(
        age=71, sex="male", chief_complaint="Chest pain",
        primary_impression="Angina", bp_systolic=120, bp_diastolic=80,
        heart_rate=80, respiratory_rate=16, spo2=98, gcs_total=15,
        avpu="Alert", allergies=["NKDA"], past_medical_history=["none"],
        medications_given=[], procedures=[], narrative_text="Test narrative"
    )
    state = _make_state(pcr)
    result = detector.detect_gaps(state)

    assert len(result.missing_mandatory) == 0


def test_batch_prompt_generation():
    detector = GapDetector()
    state = _make_state(PCRDocument())
    result = detector.detect_gaps(state)
    batch_prompt = detector.generate_batch_prompt(result)

    assert "mandatory" in batch_prompt.lower() or "required" in batch_prompt.lower()
