"""Tests for PCR State Manager."""

from app.core.pcr_state_manager import PCRStateManager
from app.schemas.pcr import PCRDocument


def test_initial_state_is_empty():
    mgr = PCRStateManager("test-session")
    state = mgr.get_state()
    assert state.session_id == "test-session"
    assert state.pcr.age is None
    assert state.version == 0
    assert state.completeness_score == 0.0


def test_apply_extraction_fills_empty_fields():
    mgr = PCRStateManager("test-session")
    extracted = PCRDocument(age=71, sex="male", heart_rate=96)
    confidence = {"age": 0.9, "sex": 0.8, "heart_rate": 0.85}

    state = mgr.apply_extraction(extracted, confidence, "test_model")

    assert state.pcr.age == 71
    assert state.pcr.sex == "male"
    assert state.pcr.heart_rate == 96
    assert state.version == 1
    assert "age" in state.field_confidence
    assert state.field_confidence["age"].confidence == 0.9


def test_extraction_rejects_low_confidence():
    mgr = PCRStateManager("test-session", confidence_threshold=0.6)
    extracted = PCRDocument(age=71, sex="male")
    confidence = {"age": 0.3, "sex": 0.8}

    state = mgr.apply_extraction(extracted, confidence, "test_model")

    assert state.pcr.age is None  # Rejected (0.3 < 0.6)
    assert state.pcr.sex == "male"  # Accepted (0.8 >= 0.6)


def test_higher_confidence_overwrites():
    mgr = PCRStateManager("test-session")

    # First extraction
    extracted1 = PCRDocument(heart_rate=96)
    mgr.apply_extraction(extracted1, {"heart_rate": 0.7}, "model_a")

    # Second extraction with higher confidence
    extracted2 = PCRDocument(heart_rate=98)
    state = mgr.apply_extraction(extracted2, {"heart_rate": 0.9}, "model_b")

    assert state.pcr.heart_rate == 98
    assert state.field_confidence["heart_rate"].confidence == 0.9


def test_lower_confidence_does_not_overwrite():
    mgr = PCRStateManager("test-session")

    extracted1 = PCRDocument(heart_rate=96)
    mgr.apply_extraction(extracted1, {"heart_rate": 0.9}, "model_a")

    extracted2 = PCRDocument(heart_rate=98)
    state = mgr.apply_extraction(extracted2, {"heart_rate": 0.6}, "model_b")

    assert state.pcr.heart_rate == 96  # Not overwritten


def test_list_field_union_merge():
    mgr = PCRStateManager("test-session")

    extracted1 = PCRDocument(allergies=["penicillin"])
    mgr.apply_extraction(extracted1, {"allergies": 0.9}, "model_a")

    extracted2 = PCRDocument(allergies=["penicillin", "sulfa"])
    state = mgr.apply_extraction(extracted2, {"allergies": 0.8}, "model_b")

    assert len(state.pcr.allergies) == 2
    assert "penicillin" in state.pcr.allergies
    assert "sulfa" in state.pcr.allergies


def test_correction_overrides_extraction():
    mgr = PCRStateManager("test-session")

    # Extract heart_rate
    extracted = PCRDocument(heart_rate=96)
    mgr.apply_extraction(extracted, {"heart_rate": 0.9}, "model_a")

    # User corrects it
    state = mgr.apply_correction("heart_rate", 108, action="update")

    assert state.pcr.heart_rate == 108
    assert state.field_confidence["heart_rate"].confidence == 1.0
    assert state.field_confidence["heart_rate"].source == "user_correction"


def test_correction_append_to_list():
    mgr = PCRStateManager("test-session")
    extracted = PCRDocument(allergies=["penicillin"])
    mgr.apply_extraction(extracted, {"allergies": 0.9}, "model_a")

    state = mgr.apply_correction("allergies", "sulfa", action="append")
    assert "sulfa" in state.pcr.allergies
    assert "penicillin" in state.pcr.allergies


def test_correction_remove_from_list():
    mgr = PCRStateManager("test-session")
    extracted = PCRDocument(allergies=["penicillin", "sulfa"])
    mgr.apply_extraction(extracted, {"allergies": 0.9}, "model_a")

    state = mgr.apply_correction("allergies", "sulfa", action="remove")
    assert "sulfa" not in state.pcr.allergies
    assert "penicillin" in state.pcr.allergies


def test_completeness_calculation():
    mgr = PCRStateManager("test-session")

    # Fill only mandatory fields
    extracted = PCRDocument(
        age=71, sex="male", chief_complaint="Chest pain",
        primary_impression="Angina"
    )
    confidence = {k: 0.9 for k in ["age", "sex", "chief_complaint", "primary_impression"]}
    state = mgr.apply_extraction(extracted, confidence, "test")

    # 4 mandatory filled out of 4 mandatory + N required = partial completeness
    assert state.completeness_score > 0.0
    assert state.completeness_score < 1.0
    assert len(state.missing_mandatory) == 0


def test_missing_fields_detection():
    mgr = PCRStateManager("test-session")
    state = mgr.get_state()

    # All mandatory should be missing initially
    assert "age" in state.missing_mandatory
    assert "sex" in state.missing_mandatory
    assert "chief_complaint" in state.missing_mandatory
    assert "primary_impression" in state.missing_mandatory

    # Required fields should also be missing
    assert "heart_rate" in state.missing_required
    assert "bp_systolic" in state.missing_required


def test_version_increments():
    mgr = PCRStateManager("test-session")
    assert mgr.get_state().version == 0

    extracted = PCRDocument(age=71)
    mgr.apply_extraction(extracted, {"age": 0.9}, "model_a")
    assert mgr.get_state().version == 1

    mgr.apply_correction("age", 72, action="update")
    assert mgr.get_state().version == 2
