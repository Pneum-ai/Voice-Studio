"""MLX catalog coverage and model-specific transcription request contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.backends.factory import BackendBundle
from app.backends.mlx.stt import MLXSTTBackend
from app.backends.mlx.tts import MLXTTSBackend
from app.backends.protocols import TranscriptionRequest
from app.core.config import Settings
from app.core.errors import ValidationError
from app.services.container import build_services


@pytest.mark.parametrize(
  ("model_id", "language", "context", "expected"),
  [
    ("qwen3-asr-0.6B", "de", "Pneum", {"language": "German", "system_prompt": "Pneum"}),
    ("qwen3-asr-1.7B", None, None, {}),
    ("whisper-base", "en", "Pneum", {"language": "en", "initial_prompt": "Pneum"}),
    (
      "nvidia-canary-1b-v2",
      "de",
      None,
      {"source_lang": "de", "target_lang": "de", "max_tokens": 8192},
    ),
    ("nvidia-parakeet-tdt-0.6b-v3", None, None, {"chunk_duration": 30.0}),
  ],
)
async def test_transcription_preserves_native_language_and_context(
  tmp_path: Path,
  model_id: str,
  language: str | None,
  context: str | None,
  expected: dict[str, object],
) -> None:
  """Keep auto-detection and hints intact across incompatible upstream model APIs."""
  backend = MLXSTTBackend(tmp_path)
  audio_path = tmp_path / "speech.wav"
  calls = []

  def generate(audio: str, **options: object) -> SimpleNamespace:
    """Capture the inference boundary and return the upstream output shape."""
    calls.append((audio, options))
    return SimpleNamespace(text="  Recognized speech.  ")

  backend._model = SimpleNamespace(generate=generate)
  backend._loaded_model = model_id
  result = await backend.transcribe(TranscriptionRequest(audio_path, language, context))
  assert calls == [(str(audio_path), expected)]
  assert result.text == "Recognized speech."
  assert result.language == language


async def test_inference_type_error_is_not_retried_without_context(tmp_path: Path) -> None:
  """Do not hide real runtime failures or silently discard a user's recognition hint."""
  backend = MLXSTTBackend(tmp_path)
  calls = []

  def generate(*args: object, **kwargs: object) -> None:
    """Represent a failure inside inference rather than argument binding."""
    calls.append(kwargs)
    raise TypeError("Invalid tensor")

  backend._model = SimpleNamespace(generate=generate)
  backend._loaded_model = "whisper-base"
  with pytest.raises(TypeError, match="Invalid tensor"):
    await backend.transcribe(TranscriptionRequest(tmp_path / "audio.wav", "en", "Pneum"))
  assert len(calls) == 1


async def test_canary_rejects_context_instead_of_silently_ignoring_it(tmp_path: Path) -> None:
  """Expose the MLX Canary context limitation before inference starts."""
  backend = MLXSTTBackend(tmp_path)
  backend._model = object()
  backend._loaded_model = "nvidia-canary-1b-v2"
  with pytest.raises(ValidationError, match="does not support recognition context"):
    await backend.transcribe(TranscriptionRequest(tmp_path / "audio.wav", "en", "Pneum"))


async def test_mlx_startup_does_not_install_pytorch_workers(
  tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
  """Start MLX without attempting incompatible, unused Qwen and NeMo installations."""
  bundle = BackendBundle("mlx", "metal", MLXTTSBackend(tmp_path), MLXSTTBackend(tmp_path))
  monkeypatch.setattr("app.services.container.create_backends", lambda *args, **kwargs: bundle)
  services = build_services(Settings(data_dir=tmp_path, prewarm=False))

  async def fail() -> None:
    """Fail if startup invokes the unnecessary runtime installer."""
    pytest.fail("MLX must not install PyTorch workers")

  monkeypatch.setattr(services.model_runtimes, "install_all", fail)
  try:
    await services.startup()
    models = await services.models.list_models()
    assert all(model.backend_supported for model in models)
    canary = services.models.definition("nvidia-canary-1b-v2")
    assert canary.transcription is not None and not canary.transcription.context
  finally:
    await services.shutdown()
