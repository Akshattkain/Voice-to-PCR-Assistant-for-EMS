"""Tests for NEMSIS schema registry."""

from app.schemas.nemsis import (
    FIELD_REGISTRY,
    NEMSISUsage,
    get_mandatory_fields,
    get_recommended_fields,
    get_required_fields,
)
from app.schemas.pcr import PCRDocument


def test_field_registry_not_empty():
    assert len(FIELD_REGISTRY) > 0


def test_all_pcr_fields_in_registry():
    """Every field in PCRDocument should have a NEMSIS registry entry."""
    pcr_fields = set(PCRDocument.model_fields.keys())
    registry_fields = set(FIELD_REGISTRY.keys())
    # All registry fields should be valid PCR fields
    assert registry_fields.issubset(pcr_fields), (
        f"Registry has fields not in PCRDocument: {registry_fields - pcr_fields}"
    )


def test_mandatory_fields_exist():
    mandatory = get_mandatory_fields()
    assert len(mandatory) >= 4  # age, sex, chief_complaint, primary_impression
    assert "age" in mandatory
    assert "sex" in mandatory
    assert "chief_complaint" in mandatory
    assert "primary_impression" in mandatory


def test_required_fields_exist():
    required = get_required_fields()
    assert "heart_rate" in required
    assert "bp_systolic" in required
    assert "allergies" in required


def test_recommended_fields_exist():
    recommended = get_recommended_fields()
    assert "pain_scale" in recommended


def test_all_fields_have_prompt_templates():
    for name, meta in FIELD_REGISTRY.items():
        assert meta.prompt_template, f"Field {name} missing prompt_template"


def test_all_fields_have_nemsis_element():
    for name, meta in FIELD_REGISTRY.items():
        assert meta.nemsis_element, f"Field {name} missing nemsis_element"


def test_usage_values_valid():
    for name, meta in FIELD_REGISTRY.items():
        assert meta.usage in NEMSISUsage, f"Field {name} has invalid usage: {meta.usage}"
