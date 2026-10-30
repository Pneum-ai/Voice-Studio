"""Deterministic fake model backends and audio fixtures."""

from __future__ import annotations

import asyncio
import io
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import numpy as np
import soundfile as sf

from app.backends.factory import BackendBundle
from app.backends.protocols import (
  BatchGenerationRequest,
  DownloadProgressCallback,
  LiveTextStream,
  TranscriptionRequest,
  TranscriptionResult,
)
from app.core.errors import ValidationError
from app.utils.audio import AudioData


class FakeLiveTextStream:
  """Provide deterministic append-triggered PCM for live protocol tests."""

  def __init__(self, cancel_event: asyncio.Event) -> None:
    """Initialize ordered text and an empty bounded output queue."""
    self._cancel_event = cancel_event
    self._started = asyncio.Event()
    self._queue: asyncio.Queue[AudioData | None] = asyncio.Queue()
    self._expected_sequence = 0
    self._text = ""

  @property
  def text(self) -> str:
    """Return the accepted fake text."""
    return self._text

  def _consume(self, sequence: int) -> None:
    """Require monotonically increasing fragment sequence numbers."""
    if sequence != self._expected_sequence:
      raise ValidationError(
        f"Expected sequence {self._expected_sequence}, got {sequence}.",
        code="invalid_sequence",
      )
    self._expected_sequence += 1

  async def append_text(self, text: str, *, sequence: int) -> str | None:
    """Start output on the first text fragment and enqueue one PCM chunk."""
    self._consume(sequence)
    self._text += text
    self._started.set()
    await self._queue.put(AudioData(np.full(1200, 0.1, dtype=np.float32), 24_000))
    return "fake-live-engine"

  async def end_text(self, *, sequence: int) -> str:
    """Close input and enqueue the final PCM and terminal markers."""
    self._consume(sequence)
    await self._queue.put(AudioData(np.full(1200, -0.1, dtype=np.float32), 24_000))
    await self._queue.put(None)
    return "fake-live-engine"

  async def wait_started(self) -> bool:
    """Wait until text arrives or cancellation is requested."""
    started = asyncio.create_task(self._started.wait())
    cancelled = asyncio.create_task(self._cancel_event.wait())
    done, pending = await asyncio.wait(
      (started, cancelled),
      return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
      task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return started in done and not self._cancel_event.is_set()

  async def _audio(self) -> AsyncIterator[AudioData]:
    """Yield queued chunks until the input closes or cancellation wins."""
    while not self._cancel_event.is_set():
      chunk = await self._queue.get()
      if chunk is None:
        return
      yield chunk

  def audio(self) -> AsyncIterator[AudioData]:
    """Return the fake live PCM iterator."""
    return self._audio()

  async def cancel(self) -> None:
    """Wake all fake waiters after cancellation."""
    self._cancel_event.set()
    self._started.set()
    await self._queue.put(None)


class FakeTTSBackend:
  """Provide deterministic Qwen-compatible behavior without model downloads."""

  def __init__(self, *, streaming: bool = True) -> None:
    """Initialize cached models with the default model loaded."""
    self._loaded_model: str | None = "qwen-tts-1.7B"
    self.cached = {"qwen-tts-0.6B", "qwen-tts-1.7B", "qwen-tts-1.7B-customvoice"}
    self.batch_sizes: list[int] = []
    self.generated_texts: list[str] = []
    self._streaming = streaming

  @property
  def loaded_model(self) -> str | None:
    """Return the fake loaded model."""
    return self._loaded_model

  @property
  def supports_streaming(self) -> bool:
    """Report native chunk support, or complete waveforms for batching tests."""
    return self._streaming

  @property
  def supports_concurrent_streaming(self) -> bool:
    """Allow fake streams to overlap for scheduler-facing service tests."""
    return self._streaming

  @property
  def supports_live_text(self) -> bool:
    """Expose fake live input only when native streaming is enabled."""
    return self._streaming

  def supports(self, model_id: str) -> bool:
    """Support both selectable Qwen model IDs."""
    return model_id in {"qwen-tts-0.6B", "qwen-tts-1.7B", "qwen-tts-1.7B-customvoice"}

  def is_cached(self, model_id: str) -> bool:
    """Return whether fake weights are present."""
    return model_id in self.cached

  def cache_size(self, model_id: str) -> int | None:
    """Return a deterministic fake size."""
    return 100 if model_id in self.cached else None

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Mark fake weights as downloaded and report deterministic bytes."""
    if progress is not None:
      await progress(100, 100)
    self.cached.add(model_id)

  async def load(self, model_id: str) -> None:
    """Mark a supported fake model as loaded."""
    await self.download(model_id)
    self._loaded_model = model_id

  async def unload(self) -> None:
    """Release the fake model."""
    self._loaded_model = None

  async def delete(self, model_id: str) -> None:
    """Remove fake weights and unload a matching model."""
    if self._loaded_model == model_id:
      self._loaded_model = None
    self.cached.discard(model_id)

  async def create_prompt(self, audio_path: Path, transcript: str) -> object:
    """Return a lightweight prompt containing reference metadata."""
    return {"ref_audio": str(audio_path), "ref_text": transcript}

  async def generate(
    self,
    *,
    text: str,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
  ) -> AudioData:
    """Return the same audio that the fake stream produces."""
    chunks = [
      chunk
      async for chunk in self.stream(
        text=text,
        prompt=prompt,
        language=language,
        seed=seed,
        instructions=instructions,
        cancel_event=asyncio.Event(),
      )
    ]
    return AudioData(
      np.concatenate([chunk.samples for chunk in chunks]),
      chunks[0].sample_rate,
    )

  async def generate_batch(
    self,
    requests: Sequence[BatchGenerationRequest],
    *,
    seed: int | None,
    cancel_events: Sequence[asyncio.Event | None] | None = None,
  ) -> list[AudioData]:
    """Record the batch size seen and return one waveform per request."""
    _ = cancel_events
    self.batch_sizes.append(len(requests))
    return [
      await self.generate(
        text=request.text,
        prompt=request.prompt,
        language=request.language,
        seed=seed,
        instructions=request.instructions,
      )
      for request in requests
    ]

  async def stream(
    self,
    *,
    text: str,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
  ) -> AsyncIterator[AudioData]:
    """Yield two deterministic PCM chunks."""
    self.generated_texts.append(text)
    del prompt, language, seed, instructions
    for value in (0.1, -0.1):
      await asyncio.sleep(0.05)
      if cancel_event.is_set():
        return
      yield AudioData(np.full(1200, value, dtype=np.float32), 24_000)

  async def open_live_text(
    self,
    *,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
  ) -> LiveTextStream:
    """Return one deterministic append-driven stream."""
    del prompt, language, seed, instructions
    if not self._streaming:
      raise RuntimeError("Fake live text is disabled.")
    return FakeLiveTextStream(cancel_event)


class FakeSTTBackend:
  """Provide deterministic Whisper-compatible behavior without weights."""

  def __init__(self) -> None:
    """Initialize the default fake Whisper model as loaded."""
    self._loaded_model: str | None = "whisper-turbo"
    self.cached = {
      "whisper-base",
      "whisper-small",
      "whisper-medium",
      "whisper-large",
      "whisper-turbo",
    }

  @property
  def loaded_model(self) -> str | None:
    """Return the loaded fake Whisper ID."""
    return self._loaded_model

  def supports(self, model_id: str) -> bool:
    """Support every selectable Whisper model."""
    return model_id in {
      "whisper-base",
      "whisper-large",
      "whisper-medium",
      "whisper-small",
      "whisper-turbo",
    }

  def is_cached(self, model_id: str) -> bool:
    """Return whether fake Whisper weights exist."""
    return model_id in self.cached

  def cache_size(self, model_id: str) -> int | None:
    """Return a deterministic fake cache size."""
    return 200 if model_id in self.cached else None

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Mark fake Whisper weights as downloaded and report deterministic bytes."""
    if progress is not None:
      await progress(200, 200)
    self.cached.add(model_id)

  async def load(self, model_id: str) -> None:
    """Load a fake Whisper model."""
    await self.download(model_id)
    self._loaded_model = model_id

  async def unload(self) -> None:
    """Release the fake Whisper model."""
    self._loaded_model = None

  async def delete(self, model_id: str) -> None:
    """Remove fake Whisper weights."""
    if self._loaded_model == model_id:
      self._loaded_model = None
    self.cached.discard(model_id)

  async def transcribe(
    self,
    request: TranscriptionRequest,
  ) -> TranscriptionResult:
    """Return deterministic text after confirming the upload exists."""
    assert request.audio_path.is_file()
    return TranscriptionResult(
      text="deterministic transcript",
      language=request.language,
    )


def fake_backends(*, streaming: bool = True) -> BackendBundle:
  """Build a complete fake backend bundle for application tests."""
  return BackendBundle(
    name="test",
    device="cpu",
    tts=FakeTTSBackend(streaming=streaming),
    stt=FakeSTTBackend(),
  )


def reference_wav_bytes(seconds: float = 2.1) -> bytes:
  """Encode a deterministic audible 24 kHz reference WAV."""
  sample_rate = 24_000
  times = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
  samples = (0.2 * np.sin(2 * np.pi * 220 * times)).astype(np.float32)
  buffer = io.BytesIO()
  sf.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
  return buffer.getvalue()
