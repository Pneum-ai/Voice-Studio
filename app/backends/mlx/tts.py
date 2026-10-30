"""MLX Qwen Base and CustomVoice backend with incremental audio emission."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from app.backends.cache import (
  cache_root,
  delete_repository,
  download_repository,
  is_repository_cached,
  repository_size,
  repository_snapshot_path,
)
from app.backends.mlx.engine import SpeechEngine
from app.backends.mlx.optimizations import optimize_model
from app.backends.mlx.worker import MLXWorker
from app.backends.protocols import (
  BatchGenerationRequest,
  DownloadProgressCallback,
  LiveTextStream,
)
from app.core.errors import ValidationError
from app.domain.speech import BuiltinVoicePrompt
from app.utils.audio import AudioData

MODEL_REPOSITORIES = {
  "qwen-tts-1.7B-customvoice": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16",
  "qwen-tts-0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
  "qwen-tts-1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
}


class MLXTTSBackend:
  """Run MLX Qwen generation and expose its native audio iterator."""

  def __init__(
    self, model_cache_dir: Path | None = None, *, worker: MLXWorker | None = None
  ) -> None:
    """Initialize an unloaded MLX backend."""
    self._worker = worker or MLXWorker()
    self._cache_root = cache_root(model_cache_dir)
    self._model: Any | None = None
    self._engine: SpeechEngine | None = None
    self._loaded_model: str | None = None
    self._lock = asyncio.Lock()

  @property
  def loaded_model(self) -> str | None:
    """Return the canonical model ID currently in memory."""
    return self._loaded_model

  @property
  def supports_streaming(self) -> bool:
    """Return true because mlx-audio exposes an audio result iterator."""
    return True

  @property
  def supports_concurrent_streaming(self) -> bool:
    """Return true because the engine owns and isolates all per-request model state."""
    return True

  @property
  def manages_request_queue(self) -> bool:
    """Bypass complete-waveform batching so saved requests join live speech immediately."""
    return True

  @property
  def supports_live_text(self) -> bool:
    """Return false because MLX requires complete text before generation."""
    return False

  async def open_live_text(
    self,
    *,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
  ) -> LiveTextStream:
    """Reject appendable input because MLX requires complete target text."""
    del prompt, language, seed, instructions, cancel_event
    raise ValidationError(
      "Live text input requires the Nari inference engine.",
      code="live_text_not_supported",
    )

  def supports(self, model_id: str) -> bool:
    """Return whether converted MLX weights exist for the model."""
    return model_id in MODEL_REPOSITORIES

  def _repository(self, model_id: str) -> str:
    """Resolve a canonical model ID to its MLX repository."""
    try:
      return MODEL_REPOSITORIES[model_id]
    except KeyError as error:
      raise ValidationError(f"Unsupported MLX Qwen model: {model_id}") from error

  def is_cached(self, model_id: str) -> bool:
    """Return whether complete MLX Qwen weights are cached."""
    return is_repository_cached(self._cache_root, self._repository(model_id))

  def cache_size(self, model_id: str) -> int | None:
    """Return cached MLX Qwen bytes."""
    return repository_size(self._cache_root, self._repository(model_id))

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Download converted MLX weights and report byte progress."""
    await download_repository(self._cache_root, self._repository(model_id), progress)

  async def load(self, model_id: str) -> None:
    """Download if necessary and load the selected MLX model."""
    async with self._lock:
      if self._loaded_model == model_id and self._model is not None:
        return
      if self._model is not None:
        await self._unload_locked()
      if not await asyncio.to_thread(self.is_cached, model_id):
        await self.download(model_id)
      model = await self._worker.run(
        self._load_sync,
        model_id,
      )
      self._model = model
      self._loaded_model = model_id
      self._engine = SpeechEngine(model, model_id, self._worker)

  def _load_sync(self, model_id: str) -> Any:
    """Load MLX Qwen through mlx-audio."""
    from mlx_audio.tts.utils import load_model  # pyright: ignore[reportMissingImports]

    snapshot = repository_snapshot_path(
      self._cache_root,
      self._repository(model_id),
    )
    model = load_model(snapshot)
    optimize_model(model)
    return model

  async def unload(self) -> None:
    """Release the loaded MLX model."""
    async with self._lock:
      await self._unload_locked()

  async def _unload_locked(self) -> None:
    """Release the loaded model while the lifecycle lock is held."""
    if self._engine is not None:
      await self._engine.close()
      self._engine = None
    await self._worker.run(self._release_sync)
    self._loaded_model = None

  def _release_sync(self) -> None:
    """Drop model state on the thread that owns its GPU streams."""
    self._model = None

  async def delete(self, model_id: str) -> None:
    """Unload the selected model and delete its converted weights."""
    if self._loaded_model == model_id:
      await self.unload()
    await delete_repository(self._cache_root, self._repository(model_id))

  async def create_prompt(self, audio_path: Path, transcript: str) -> object:
    """Return the path-and-transcript prompt consumed by mlx-audio."""
    if self._model is None:
      raise RuntimeError("MLX Qwen model is not loaded.")
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
    """Collect streaming MLX chunks into one waveform for HTTP generation."""
    cancel_event = asyncio.Event()
    chunks = [
      chunk
      async for chunk in self.stream(
        text=text,
        prompt=prompt,
        language=language,
        seed=seed,
        instructions=instructions,
        cancel_event=cancel_event,
      )
    ]
    if not chunks:
      return AudioData(np.array([], dtype=np.float32), 24_000)
    sample_rate = chunks[0].sample_rate
    return AudioData(
      np.concatenate([chunk.samples for chunk in chunks]).astype(np.float32),
      sample_rate,
    )

  async def generate_batch(
    self,
    requests: Sequence[BatchGenerationRequest],
    *,
    seed: int | None,
    cancel_events: Sequence[asyncio.Event | None] | None = None,
  ) -> list[AudioData]:
    """Collect the same stream used by live speech, preserving caller order and cancellation."""
    events = list(cancel_events) if cancel_events is not None else [None] * len(requests)
    if len(events) != len(requests):
      raise ValueError("Cancellation events must match the request count.")

    async def collect(request: BatchGenerationRequest, event: asyncio.Event | None) -> AudioData:
      """Collect one request without a second native generation implementation."""
      chunks = [
        chunk
        async for chunk in self.stream(
          text=request.text,
          prompt=request.prompt,
          language=request.language,
          seed=seed,
          instructions=request.instructions,
          cancel_event=event or asyncio.Event(),
        )
      ]
      if event is not None and event.is_set():
        return AudioData(np.empty(0, dtype=np.float32), 24000)
      return AudioData(
        np.concatenate([chunk.samples for chunk in chunks]) if chunks else np.empty(0, np.float32),
        24000,
      )

    tasks = [
      asyncio.create_task(collect(request, event)) for request, event in zip(requests, events)
    ]
    try:
      return list(await asyncio.gather(*tasks))
    finally:
      for task in tasks:
        if not task.done():
          task.cancel()
      await asyncio.gather(*tasks, return_exceptions=True)

  async def stream(
    self,
    *,
    text: str,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
  ) -> AsyncGenerator[AudioData, None]:
    """Submit live and saved speech to one bounded, continuously batched engine."""
    if self._engine is None:
      raise RuntimeError("MLX Qwen model is not loaded.")
    speaker = prompt.speaker if isinstance(prompt, BuiltinVoicePrompt) else None
    if speaker is not None:
      reference, transcript = None, None
    elif isinstance(prompt, dict):
      reference, transcript = prompt.get("ref_audio"), prompt.get("ref_text")
      if not isinstance(reference, str) or not isinstance(transcript, str):
        raise ValidationError("MLX cloning requires a reference audio path and transcript.")
    else:
      raise TypeError("MLX speech requires a reference or built-in speaker prompt.")
    iterator = self._engine.stream(
      text=text,
      reference=reference,
      speaker=speaker,
      transcript=transcript,
      language=language,
      instructions=instructions,
      seed=seed,
      cancel=cancel_event,
    )
    try:
      async for audio in iterator:
        yield audio
    finally:
      await iterator.aclose()
