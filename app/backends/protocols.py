"""Typed backend contracts consumed by model and speech services."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from app.utils.audio import AudioData, StreamAudioData

DownloadProgressCallback = Callable[[int, int | None], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class BatchGenerationRequest:
  """Hold one resolved generation that may share a pass with other requests."""

  text: str
  prompt: object
  language: str
  instructions: str | None


@dataclass(frozen=True, slots=True)
class TranscriptionRequest:
  """Hold normalized, model-independent speech recognition options."""

  audio_path: Path
  language: str | None
  context: str | None


@dataclass(frozen=True, slots=True)
class TranscriptionResult:
  """Return normalized recognition text and the model-reported language."""

  text: str
  language: str | None = None


class ModelBackend(Protocol):
  """Provide common local model lifecycle operations."""

  @property
  def loaded_model(self) -> str | None:
    """Return the canonical ID of the model currently held in memory."""
    ...

  def supports(self, model_id: str) -> bool:
    """Return whether this backend can run the canonical model ID."""
    ...

  def is_cached(self, model_id: str) -> bool:
    """Return whether complete model weights are present locally."""
    ...

  def cache_size(self, model_id: str) -> int | None:
    """Return local model bytes when available."""
    ...

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Download a model and optionally report current and total bytes."""
    ...

  async def load(self, model_id: str) -> None:
    """Load a locally cached model into memory."""
    ...

  async def unload(self) -> None:
    """Release the currently loaded model."""
    ...

  async def delete(self, model_id: str) -> None:
    """Remove locally cached model files."""
    ...


class TTSBackend(ModelBackend, Protocol):
  """Generate cloned speech from reference prompts."""

  @property
  def supports_streaming(self) -> bool:
    """Return whether audio can be emitted during model generation."""
    ...

  @property
  def supports_concurrent_streaming(self) -> bool:
    """Return whether independent streams may share the loaded model safely."""
    ...

  @property
  def supports_live_text(self) -> bool:
    """Return whether text can be appended while audio generation is active."""
    ...

  async def create_prompt(self, audio_path: Path, transcript: str) -> object:
    """Create a backend-specific voice-clone prompt."""
    ...

  async def generate(
    self,
    *,
    text: str,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
  ) -> AudioData:
    """Generate one complete waveform."""
    ...

  async def generate_batch(
    self,
    requests: Sequence[BatchGenerationRequest],
    *,
    seed: int | None,
    cancel_events: Sequence[asyncio.Event | None] | None = None,
  ) -> list[AudioData]:
    """Generate one waveform per request, stopping once every caller cancels."""
    ...

  def stream(
    self,
    *,
    text: str,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
  ) -> AsyncIterator[StreamAudioData]:
    """Yield audio as backend chunks become available."""
    ...

  async def open_live_text(
    self,
    *,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
  ) -> LiveTextStream:
    """Open one appendable text-to-audio request."""
    ...


class LiveTextStream(Protocol):
  """Accept ordered text fragments while exposing one incremental PCM stream."""

  @property
  def text(self) -> str:
    """Return all text accepted by the live request."""
    ...

  async def append_text(self, text: str, *, sequence: int) -> str | None:
    """Append text and return the engine request ID once synthesis starts."""
    ...

  async def end_text(self, *, sequence: int) -> str:
    """Flush the mutable tokenizer tail and close text input."""
    ...

  async def wait_started(self) -> bool:
    """Wait until synthesis starts, returning false if cancelled first."""
    ...

  def audio(self) -> AsyncIterator[StreamAudioData]:
    """Yield audio until generation completes or is cancelled."""
    ...

  async def cancel(self) -> None:
    """Cancel the live engine request when it has been submitted."""
    ...


class STTBackend(ModelBackend, Protocol):
  """Transcribe audio with a selected local model."""

  async def transcribe(
    self,
    request: TranscriptionRequest,
  ) -> TranscriptionResult:
    """Return normalized transcription text and detected language metadata."""
    ...
