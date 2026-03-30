"""Validate generated (transcript, PCR JSON) pairs for schema compliance."""

from app.schemas.pcr import PCRDocument
from app.utils.logging import logger
from training.data_gen.synthetic_generator import DataPair


def validate_pair(pair: DataPair) -> tuple[bool, list[str]]:
    """Validate a single data pair. Returns (is_valid, list of errors)."""
    errors = []

    # Check transcript is non-empty
    if not pair.transcript or len(pair.transcript.strip()) < 20:
        errors.append("Transcript is empty or too short")

    # Check PCR JSON is non-empty
    if not pair.pcr_json:
        errors.append("PCR JSON is empty")
        return False, errors

    # Validate PCR JSON against schema
    try:
        pcr = PCRDocument(**pair.pcr_json)
    except Exception as e:
        errors.append(f"PCR JSON schema validation failed: {e}")
        return False, errors

    # Check that mandatory fields are present (unless difficulty=incomplete)
    if pair.difficulty != "incomplete":
        if pcr.age is None:
            errors.append("Missing mandatory field: age")
        if pcr.sex is None:
            errors.append("Missing mandatory field: sex")
        if pcr.chief_complaint is None:
            errors.append("Missing mandatory field: chief_complaint")

    # Check that at least some fields have values
    pcr_dict = pcr.model_dump(exclude_none=True)
    non_list_fields = {k: v for k, v in pcr_dict.items() if not isinstance(v, list)}
    if len(non_list_fields) < 3:
        errors.append("Too few fields populated (need at least 3 non-list fields)")

    return len(errors) == 0, errors


def validate_batch(pairs: list[DataPair]) -> dict:
    """Validate a batch of pairs. Returns summary statistics."""
    valid_count = 0
    invalid_count = 0
    all_errors = []

    for i, pair in enumerate(pairs):
        is_valid, errors = validate_pair(pair)
        if is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            all_errors.append({"index": i, "errors": errors})

    return {
        "total": len(pairs),
        "valid": valid_count,
        "invalid": invalid_count,
        "validation_rate": valid_count / len(pairs) if pairs else 0,
        "errors": all_errors,
    }
