"""PyTorch Qwen Base and CustomVoice backend."""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from app.backends.cache import (
  cache_root,
  delete_repository,
  download_repository,
  is_repository_cached,
  repository_size,
  repository_snapshot_path,
)
from app.backends.protocols import BatchGenerationRequest, DownloadProgressCallback
from app.backends.pytorch.device import select_device
from app.backends.pytorch.nari_live_backend import NariLiveBackendMixin
from app.backends.pytorch.nari_streaming import NariRuntimeConfig
from app.backends.pytorch.qwen_generation import (
  LANGUAGE_NAMES,
  cancelable_qwen_generation,
  qwen_generation_token_limit,
)
from app.core.errors import ValidationError
from app.domain.speech import BuiltinVoicePrompt
from app.utils.async_work import run_in_daemon_thread
from app.utils.audio import AudioData, StreamAudioData

MODEL_REPOSITORIES = {
  "qwen-tts-1.7B-customvoice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
  "qwen-tts-1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "qwen-tts-0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
LOGGER = logging.getLogger(__name__)


class PyTorchTTSBackend(NariLiveBackendMixin):
  """Run Qwen synthesis through Nari or the upstream PyTorch wrapper."""

  def __init__(
    self,
    model_cache_dir: Path | None = None,
    *,
    compile_model: bool = False,
    engine: str = "nari",
    nari_config: NariRuntimeConfig | None = None,
  ) -> None:
    """Initialize an unloaded backend and select its compute device."""
    import torch

    self._torch = torch
    self._cache_root = cache_root(model_cache_dir)
    self._device = select_device(torch)
    self._compile_model = compile_model
    if engine not in {"auto", "nari", "official"}:
      raise ValueError("Qwen engine must be auto, nari, or official.")
    self._engine = engine
    self._nari_config = nari_config or NariRuntimeConfig()
    self._nari: Any | None = None
    self._model: Any | None = None
    self._loaded_model: str | None = None
    self._lock = asyncio.Lock()

  @property
  def loaded_model(self) -> str | None:
    """Return the canonical model ID currently in memory."""
    return self._loaded_model

  @property
  def supports_streaming(self) -> bool:
    """Return whether the selected runtime emits incremental Codec audio."""
    return self._nari_runtime() is not None

  @property
  def supports_concurrent_streaming(self) -> bool:
    """Return true when Nari's unified scheduler owns request concurrency."""
    return self._nari_runtime() is not None

  @property
  def supports_live_text(self) -> bool:
    """Return true when Nari can accept text after generation starts."""
    return self._nari_runtime() is not None

  @property
  def runtime_status(self) -> object | None:
    """Return Nari readiness and pressure when the accelerated runtime is active."""
    nari = self._nari_runtime()
    return nari.runtime_status() if nari is not None else None

  def _nari_runtime(self) -> Any | None:
    """Return Nari safely for backend instances created by older test fixtures."""
    return getattr(self, "_nari", None)

  def supports(self, model_id: str) -> bool:
    """Return whether the model is a supported Qwen speech model."""
    return model_id in MODEL_REPOSITORIES

  def _repository(self, model_id: str) -> str:
    """Resolve a canonical model ID to its Hugging Face repository."""
    try:
      return MODEL_REPOSITORIES[model_id]
    except KeyError as error:
      raise ValidationError(f"Unsupported Qwen model: {model_id}") from error

  def is_cached(self, model_id: str) -> bool:
    """Return whether complete Qwen weights are cached locally."""
    return is_repository_cached(self._cache_root, self._repository(model_id))

  def cache_size(self, model_id: str) -> int | None:
    """Return cached Qwen bytes."""
    return repository_size(self._cache_root, self._repository(model_id))

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Download Qwen weights without loading them and report byte progress."""
    await download_repository(self._cache_root, self._repository(model_id), progress)

  async def load(self, model_id: str) -> None:
    """Download if necessary and load the selected Qwen model."""
    async with self._lock:
      nari = self._nari_runtime()
      if self._loaded_model == model_id and (
        self._model is not None or (nari is not None and nari.runtime_status().ready)
      ):
        return
      if self._model is not None or self._nari_runtime() is not None:
        await self._unload_locked()
      if not await asyncio.to_thread(self.is_cached, model_id):
        await self.download(model_id)
      if self._engine != "official":
        try:
          self._nari = await run_in_daemon_thread(
            self._load_nari_sync,
            model_id,
            thread_name="voice-qwen-nari-load",
          )
        except Exception as error:
          if self._engine == "nari":
            raise
          LOGGER.warning("Nari Qwen runtime unavailable; using official runtime: %s", error)
      if self._nari_runtime() is None:
        self._model = await run_in_daemon_thread(
          self._load_sync,
          model_id,
          thread_name="voice-qwen-load",
        )
      self._loaded_model = model_id

  def _load_nari_sync(
    self,
    model_id: str,
  ) -> Any:
    """Load Nari before graph capture."""
    from app.backends.pytorch.nari_tts import NariTTSRuntime

    available, reason = NariTTSRuntime.supported(device=str(self._device))
    if not available:
      raise RuntimeError(reason)
    repository = repository_snapshot_path(self._cache_root, self._repository(model_id))
    return NariTTSRuntime(
      repository,
      device=str(self._device),
      model_id=model_id,
      config=self._nari_config,
    )

  def _load_sync(self, model_id: str) -> Any:
    """Load Qwen through its blocking upstream API."""
    from app.backends.pytorch.qwen_runtime import Qwen3TTSModel

    torch = self._torch
    repository = repository_snapshot_path(
      self._cache_root,
      self._repository(model_id),
    )
    dtype = torch.float32 if self._device == "cpu" else torch.bfloat16
    kwargs: dict[str, Any] = {
      "dtype": dtype,
      "low_cpu_mem_usage": False,
    }
    if self._device != "cpu":
      kwargs["device_map"] = self._device
    model = Qwen3TTSModel.from_pretrained(str(repository), **kwargs)
    if self._device == "cuda" and self._compile_model:
      try:
        capability = torch.cuda.get_device_capability()
        mode = "max-autotune" if capability[0] >= 9 else "reduce-overhead"
        model_wrapper = cast(Any, model)
        model_wrapper.model = torch.compile(model_wrapper.model, mode=mode)
      except Exception as error:
        LOGGER.warning("Qwen torch.compile was skipped: %s", error)
    return model

  async def unload(self) -> None:
    """Release Qwen and clear accelerator caches."""
    async with self._lock:
      await self._unload_locked()

  async def _unload_locked(self) -> None:
    """Release Qwen while the lifecycle lock is held."""
    nari = self._nari_runtime()
    self._nari = None
    if nari is not None:
      await run_in_daemon_thread(nari.stop, thread_name="voice-qwen-nari-stop")
    self._model = None
    self._loaded_model = None
    if self._torch.cuda.is_available():
      self._torch.cuda.empty_cache()

  async def delete(self, model_id: str) -> None:
    """Unload the selected Qwen model and remove its cache."""
    if self._loaded_model == model_id:
      await self.unload()
    await delete_repository(self._cache_root, self._repository(model_id))

  async def create_prompt(self, audio_path: Path, transcript: str) -> object:
    """Create a Qwen voice-clone prompt from normalized reference audio."""
    nari = self._nari_runtime()
    if nari is not None:
      return await run_in_daemon_thread(
        nari.create_prompt,
        audio_path,
        transcript,
        thread_name="voice-qwen-nari-prompt",
      )
    model = self._model
    if model is None:
      raise RuntimeError("Qwen model is not loaded.")

    def create() -> object:
      """Run blocking prompt creation."""
      return model.create_voice_clone_prompt(
        ref_audio=str(audio_path),
        ref_text=transcript,
        x_vector_only_mode=False,
      )

    return await run_in_daemon_thread(
      create,
      thread_name="voice-qwen-prompt",
    )

  async def generate(
    self,
    *,
    text: str,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
  ) -> AudioData:
    """Generate one complete Qwen waveform outside the event loop."""
    generated = await self.generate_batch(
      [
        BatchGenerationRequest(
          text=text,
          prompt=prompt,
          language=language,
          instructions=instructions,
        )
      ],
      seed=seed,
    )
    return generated[0]

  async def generate_batch(
    self,
    requests: Sequence[BatchGenerationRequest],
    *,
    seed: int | None,
    cancel_events: Sequence[asyncio.Event | None] | None = None,
  ) -> list[AudioData]:
    """Generate every request in one batched Qwen pass outside the event loop.

    Qwen decodes 16 sub-talker codebooks per 12 Hz frame, so a single request
    is bound by kernel launches rather than compute. Batching shares those
    launches and is what keeps generation ahead of real time.
    """
    if not requests:
      return []
    nari = self._nari_runtime()
    if nari is not None:
      return await nari.generate_batch(
        requests,
        seed=seed,
        cancel_events=cancel_events,
      )
    model = self._model
    if model is None:
      raise RuntimeError("Qwen model is not loaded.")
    token_limit = max(qwen_generation_token_limit(request.text) for request in requests)
    cancel_signal = threading.Event()

    def generate_sync() -> list[AudioData]:
      """Run blocking batched autoregressive generation and codec decoding."""
      if seed is not None:
        self._torch.manual_seed(seed)
        if self._torch.cuda.is_available():
          self._torch.cuda.manual_seed_all(seed)
      prompt_items: list[Any] = []
      for request in requests:
        if isinstance(request.prompt, list):
          prompt_items.extend(request.prompt)
        else:
          prompt_items.append(request.prompt)
      if len(prompt_items) != len(requests):
        raise RuntimeError("Each batched request must resolve to exactly one voice prompt.")
      kwargs: dict[str, Any] = {
        "text": [request.text for request in requests],
        "language": [
          LANGUAGE_NAMES.get(request.language.casefold(), request.language) for request in requests
        ],
        "voice_clone_prompt": prompt_items,
        "max_new_tokens": token_limit,
      }
      custom = isinstance(requests[0].prompt, BuiltinVoicePrompt)
      generate = model.generate_voice_clone
      if custom:
        if not all(isinstance(item, BuiltinVoicePrompt) for item in prompt_items):
          raise ValidationError("Cannot mix cloned and built-in voices in one model batch.")
        kwargs.pop("voice_clone_prompt")
        kwargs["speaker"] = [item.speaker for item in prompt_items]
        kwargs["instruct"] = [request.instructions or "" for request in requests]
        generate = model.generate_custom_voice
      elif any(request.instructions for request in requests):
        raise ValidationError("Base models do not support instructions.")
      with (
        self._torch.inference_mode(),
        cancelable_qwen_generation(model, cancel_signal),
      ):
        waveforms, sample_rate = generate(**kwargs)
      return [
        AudioData(np.asarray(waveform, dtype=np.float32), int(sample_rate))
        for waveform in waveforms
      ]

    normalized_cancel_events = tuple(cancel_events or ())
    can_cancel_batch = len(normalized_cancel_events) == len(requests) and all(
      event is not None for event in normalized_cancel_events
    )

    async def mirror_cancellation() -> None:
      """Set the thread-safe stop signal after every batched caller cancels."""
      events = [event for event in normalized_cancel_events if event is not None]
      await asyncio.gather(*(event.wait() for event in events))
      cancel_signal.set()

    cancellation_task: asyncio.Task[None] | None = None
    if can_cancel_batch:
      if all(event is not None and event.is_set() for event in normalized_cancel_events):
        cancel_signal.set()
      else:
        cancellation_task = asyncio.create_task(
          mirror_cancellation(),
          name="voice-qwen-cancellation",
        )
    try:
      return await run_in_daemon_thread(
        generate_sync,
        thread_name="voice-qwen-generate",
      )
    finally:
      if can_cancel_batch and all(
        event is not None and event.is_set() for event in normalized_cancel_events
      ):
        cancel_signal.set()
      if cancellation_task is not None:
        cancellation_task.cancel()
        await asyncio.gather(cancellation_task, return_exceptions=True)

  async def stream(
    self,
    *,
    text: str,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
  ) -> AsyncIterator[StreamAudioData]:
    """Yield native Nari chunks or one official-runtime waveform."""
    nari = self._nari_runtime()
    if nari is not None:
      request = BatchGenerationRequest(
        text=text,
        prompt=prompt,
        language=language,
        instructions=instructions,
      )
      async for chunk in nari.stream(
        request,
        seed=seed,
        cancel_event=cancel_event,
      ):
        yield chunk
      return
    generated = await self.generate_batch(
      [
        BatchGenerationRequest(
          text=text,
          prompt=prompt,
          language=language,
          instructions=instructions,
        )
      ],
      seed=seed,
      cancel_events=[cancel_event],
    )
    if not cancel_event.is_set():
      yield generated[0]
