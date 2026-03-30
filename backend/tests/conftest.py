"""Shared test fixtures."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.schemas.pcr import PCRDocument

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def sample_transcript() -> str:
    return (FIXTURES_DIR / "sample_transcript.txt").read_text().strip()


@pytest.fixture
def sample_pcr() -> PCRDocument:
    data = json.loads((FIXTURES_DIR / "sample_pcr.json").read_text())
    return PCRDocument(**data)


@pytest.fixture
def sample_pcr_dict() -> dict:
    return json.loads((FIXTURES_DIR / "sample_pcr.json").read_text())
