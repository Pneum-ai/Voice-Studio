"""Transcription model-family registry lifecycle tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.backends.protocols import (
  DownloadProgressCallback,
  TranscriptionRequest,
  TranscriptionResult,
)
from app.backends.stt_registry import STTRegistry


class FakeAdapter:
  """Provide one deterministic model-family adapter for registry tests."""

  def __init__(self, model_ids: set[str], label: str) -> None:
    """Store supported IDs and initialize empty lifecycle state."""
    self.model_ids = model_ids
    self.label = label
    self._loaded_model: str | None = None
    self.cached: set[str] = set()
    self.unload_count = 0

  @property
  def loaded_model(self) -> str | None:
    """Return the loaded fake model ID."""
    return self._loaded_model

  def supports(self, model_id: str) -> bool:
    """Return whether this family owns the model ID."""
    return model_id in self.model_ids

  def is_cached(self, model_id: str) -> bool:
    """Return whether fake weights exist."""
    return model_id in self.cached

  def cache_size(self, model_id: str) -> int | None:
    """Return a stable fake size for downloaded models."""
    return 10 if model_id in self.cached else None

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Mark a fake model as downloaded and report deterministic bytes."""
    if progress is not None:
      await progress(10, 10)
    self.cached.add(model_id)

  async def load(self, model_id: str) -> None:
    """Load one supported fake model."""
    await self.download(model_id)
    self._loaded_model = model_id

  async def unload(self) -> None:
    """Release the loaded fake model."""
    if self._loaded_model is not None:
      self.unload_count += 1
    self._loaded_model = None

  async def delete(self, model_id: str) -> None:
    """Remove one fake model."""
    if self._loaded_model == model_id:
      await self.unload()
    self.cached.discard(model_id)

  async def transcribe(
    self,
    request: TranscriptionRequest,
  ) -> TranscriptionResult:
    """Return a family-labelled deterministic transcript."""
    return TranscriptionResult(text=self.label, language=request.language)


@pytest.mark.asyncio
async def test_loading_another_family_unloads_the_previous_adapter() -> None:
  """Guarantee the existing one-loaded-STT-model memory policy."""
  whisper = FakeAdapter({"whisper"}, "whisper")
  parakeet = FakeAdapter({"parakeet"}, "parakeet")
  registry = STTRegistry((whisper, parakeet))

  await registry.load("whisper")
  await registry.load("parakeet")

  assert whisper.loaded_model is None
  assert whisper.unload_count == 1
  assert registry.loaded_model == "parakeet"


@pytest.mark.asyncio
async def test_transcription_dispatches_to_the_loaded_family(tmp_path: Path) -> None:
  """Use active model ownership instead of Whisper-specific routing."""
  whisper = FakeAdapter({"whisper"}, "whisper")
  qwen = FakeAdapter({"qwen"}, "qwen")
  registry = STTRegistry((whisper, qwen))
  audio = tmp_path / "audio.wav"
  audio.touch()
  await registry.load("qwen")

  result = await registry.transcribe(
    TranscriptionRequest(audio_path=audio, language="en", context="Pneum")
  )

  assert result == TranscriptionResult(text="qwen", language="en")
