"""Tests for evaluation metrics."""

from app.schemas.pcr import PCRDocument
from evaluation.metrics import (
    compute_aggregate_f1,
    compute_completeness,
    compute_field_f1,
    compute_hallucination_rate,
)


def _make_full_pcr():
    return PCRDocument(
        age=71, sex="male", chief_complaint="Chest Pain/Discomfort",
        primary_impression="Angina pectoris, unspecified",
        bp_systolic=158, bp_diastolic=94, heart_rate=96,
        respiratory_rate=18, spo2=94, gcs_total=15, avpu="Alert",
        pain_scale=8, allergies=["penicillin"],
        medications_current=["metoprolol", "aspirin"],
        past_medical_history=["CAD", "hypertension"],
        medications_given=[
            {"drug": "Aspirin", "dose": 324.0, "unit": "mg", "route": "PO"},
        ],
        procedures=["12 lead electrocardiogram"],
        signs_symptoms=["Chest pain, unspecified"],
    )


def test_perfect_match_f1():
    pcr = _make_full_pcr()
    scores = compute_field_f1(pcr, pcr)
    for field, score in scores.items():
        assert score.f1 == 1.0, f"Field {field} should have F1=1.0"


def test_empty_prediction_f1():
    pred = PCRDocument()
    gt = _make_full_pcr()
    scores = compute_field_f1(pred, gt)
    # Fields in gt but not pred should have F1=0
    for field, score in scores.items():
        assert score.f1 == 0.0, f"Field {field} should have F1=0.0"


def test_numeric_tolerance():
    pred = PCRDocument(heart_rate=97)  # Off by 1
    gt = PCRDocument(heart_rate=96)
    scores = compute_field_f1(pred, gt)
    assert scores["heart_rate"].f1 == 1.0  # Within tolerance of 2


def test_numeric_out_of_tolerance():
    pred = PCRDocument(heart_rate=100)  # Off by 4
    gt = PCRDocument(heart_rate=96)
    scores = compute_field_f1(pred, gt)
    assert scores["heart_rate"].f1 == 0.0  # Outside tolerance of 2


def test_string_partial_match():
    pred = PCRDocument(chief_complaint="Chest Pain")
    gt = PCRDocument(chief_complaint="Chest Pain/Discomfort")
    scores = compute_field_f1(pred, gt)
    # Should have partial F1 (tokens overlap)
    assert 0 < scores["chief_complaint"].f1 < 1.0


def test_list_f1_perfect():
    pred = PCRDocument(allergies=["penicillin", "sulfa"])
    gt = PCRDocument(allergies=["penicillin", "sulfa"])
    scores = compute_field_f1(pred, gt)
    assert scores["allergies"].f1 == 1.0


def test_list_f1_partial():
    pred = PCRDocument(allergies=["penicillin"])
    gt = PCRDocument(allergies=["penicillin", "sulfa"])
    scores = compute_field_f1(pred, gt)
    assert scores["allergies"].precision == 1.0
    assert scores["allergies"].recall == 0.5


def test_aggregate_f1():
    pred = _make_full_pcr()
    scores = compute_field_f1(pred, pred)
    agg = compute_aggregate_f1(scores)
    assert agg["macro_f1"] == 1.0
    assert agg["micro_f1"] == 1.0


def test_hallucination_none_for_perfect():
    pcr = _make_full_pcr()
    transcript = "71-year-old male chest pain angina BP 158 over 94 HR 96 RR 18 sats 94 GCS 15 alert pain 8 penicillin metoprolol aspirin CAD hypertension Aspirin 324 mg PO 12 lead electrocardiogram"
    result = compute_hallucination_rate(pcr, pcr, transcript)
    assert result.hallucination_rate == 0.0


def test_hallucination_detected():
    pred = PCRDocument(age=71, sex="male", temperature=98.6)  # temp not in transcript
    gt = PCRDocument(age=71, sex="male")
    transcript = "71-year-old male"
    result = compute_hallucination_rate(pred, gt, transcript)
    assert result.hallucination_count >= 1
    assert "temperature" in result.hallucinated_fields


def test_completeness_empty():
    pcr = PCRDocument()
    result = compute_completeness(pcr)
    assert result.overall_completeness == 0.0
    assert result.mandatory_filled == 0
    assert len(result.missing_mandatory) > 0


def test_completeness_full():
    pcr = PCRDocument(
        age=71, sex="male", chief_complaint="CP",
        primary_impression="Angina", bp_systolic=120, bp_diastolic=80,
        heart_rate=80, respiratory_rate=16, spo2=98, gcs_total=15,
        avpu="Alert", allergies=["NKDA"], past_medical_history=["none"],
        medications_given=[], procedures=[], narrative_text="Test"
    )
    result = compute_completeness(pcr)
    assert result.mandatory_rate == 1.0
    assert result.overall_completeness > 0.8
