"""Model-specific transcription option validation tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from app.backends.pytorch.nemo_stt_worker import _transcribe, _transcription_input
from app.core.config import Settings
from app.core.errors import ValidationError
from app.services.jobs import JobService
from app.services.models import ModelService
from app.services.transcriptions import TranscriptionService
from tests.helpers import fake_backends


def transcription_service() -> TranscriptionService:
  """Build a deterministic service without downloading model weights."""
  backends = fake_backends()
  models = ModelService(Settings(prewarm=False), backends, JobService(300))
  return TranscriptionService(backends, models)


@pytest.mark.asyncio
async def test_parakeet_rejects_language_and_context_controls(tmp_path: Path) -> None:
  """Do not silently pretend automatic Parakeet supports recognition hints."""
  service = transcription_service()
  audio = tmp_path / "unused.wav"

  with pytest.raises(ValidationError, match="detects language automatically"):
    await service.transcribe(
      audio_path=audio,
      model="nvidia-parakeet-tdt-0.6b-v3",
      language="en",
      initial_prompt=None,
    )
  with pytest.raises(ValidationError, match="does not support recognition context"):
    await service.transcribe(
      audio_path=audio,
      model="nvidia-parakeet-tdt-0.6b-v3",
      language=None,
      initial_prompt="Pneum",
    )


@pytest.mark.asyncio
async def test_canary_requires_a_supported_source_language(tmp_path: Path) -> None:
  """Require Canary prompt tokens before attempting model readiness checks."""
  with pytest.raises(ValidationError, match="requires a language"):
    await transcription_service().transcribe(
      audio_path=tmp_path / "unused.wav",
      model="nvidia-canary-1b-v2",
      language=None,
      initial_prompt=None,
    )


def test_canary_worker_uses_the_v2_transcription_signature() -> None:
  """Send only the language arguments documented by Canary 1B v2."""

  class Output:
    """Represent one NeMo hypothesis."""

    text = "Canary transcript"
    language = "en"

  class Canary:
    """Reject legacy keyword arguments through an exact method signature."""

    def transcribe(
      self,
      audio_paths: list[str],
      *,
      source_lang: str,
      target_lang: str,
    ) -> list[Output]:
      """Return one deterministic hypothesis."""
      assert audio_paths == ["sample.wav"]
      assert source_lang == "en"
      assert target_lang == "en"
      return [Output()]

  result = _transcribe(Canary(), "nvidia-canary-1b-v2", "sample.wav", "en", None)

  assert result == {"ok": True, "text": "Canary transcript", "language": "en"}


def test_nemo_worker_uses_memory_audio_on_windows(tmp_path: Path, monkeypatch) -> None:
  """Bypass NeMo's locked temporary manifest for single-file transcription."""
  audio = tmp_path / "sample.wav"
  sf.write(audio, np.zeros(8_000, dtype=np.float32), 8_000)
  model = SimpleNamespace(preprocessor=SimpleNamespace(_sample_rate=16_000))
  monkeypatch.setattr(
    "app.backends.pytorch.nemo_stt_worker.sys",
    SimpleNamespace(platform="win32"),
  )

  transcription_input = _transcription_input(model, str(audio))

  assert isinstance(transcription_input, np.ndarray)
  assert transcription_input.dtype == np.float32
  assert transcription_input.shape == (16_000,)
