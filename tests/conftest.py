"""Shared isolated application fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.helpers import fake_backends, reference_wav_bytes


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
  """Run the API with isolated storage and deterministic fake inference."""
  settings = Settings(
    data_dir=tmp_path / "data",
    model_cache_dir=tmp_path / "models",
    prewarm=False,
  )
  application = create_app(settings, fake_backends())
  with TestClient(application) as test_client:
    yield test_client


@pytest.fixture
def batching_client(tmp_path: Path) -> Iterator[TestClient]:
  """Run the API with a complete-waveform backend so requests are batched."""
  settings = Settings(
    data_dir=tmp_path / "data",
    model_cache_dir=tmp_path / "models",
    prewarm=False,
    batch_max_size=4,
    batch_window_ms=500,
  )
  application = create_app(settings, fake_backends(streaming=False))
  with TestClient(application) as test_client:
    yield test_client


@pytest.fixture
def profile_with_sample(client: TestClient) -> dict[str, object]:
  """Create a profile containing one valid normalized reference sample."""
  profile = client.post("/v1/profiles", json={"name": "Test voice"}).json()
  response = client.post(
    f"/v1/profiles/{profile['id']}/samples",
    files={"audio": ("reference.wav", reference_wav_bytes())},
    data={"transcript": "This is the exact reference transcript."},
  )
  assert response.status_code == 201, response.text
  return profile
