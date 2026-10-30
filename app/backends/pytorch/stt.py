"""PyTorch Whisper transcription backend."""

from __future__ import annotations

import asyncio
import inspect
import logging
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
from app.backends.protocols import (
  DownloadProgressCallback,
  TranscriptionRequest,
  TranscriptionResult,
)
from app.backends.pytorch.device import select_device
from app.core.errors import ValidationError
from app.utils.async_work import run_in_daemon_thread
from app.utils.audio import load_audio

MODEL_REPOSITORIES = {
  "whisper-base": "openai/whisper-base",
  "whisper-small": "openai/whisper-small",
  "whisper-medium": "openai/whisper-medium",
  "whisper-large": "openai/whisper-large",
  "whisper-turbo": "openai/whisper-large-v3-turbo",
}
LOGGER = logging.getLogger(__name__)


class PyTorchSTTBackend:
  """Run Whisper transcription through Transformers."""

  def __init__(self, model_cache_dir: Path | None = None) -> None:
    """Initialize an unloaded backend and select its compute device."""
    import torch

    self._torch = torch
    self._cache_root = cache_root(model_cache_dir)
    self._device = select_device(torch)
    self._model: Any | None = None
    self._processor: Any | None = None
    self._loaded_model: str | None = None
    self._model_dtype: Any = torch.float32
    self._generate_parameters: set[str] = set()
    self._generate_accepts_kwargs = False
    self._lock = asyncio.Lock()

  @property
  def loaded_model(self) -> str | None:
    """Return the canonical Whisper model ID currently in memory."""
    return self._loaded_model

  def supports(self, model_id: str) -> bool:
    """Return whether this backend supports the Whisper model."""
    return model_id in MODEL_REPOSITORIES

  def _repository(self, model_id: str) -> str:
    """Resolve a canonical model ID to its Hugging Face repository."""
    try:
      return MODEL_REPOSITORIES[model_id]
    except KeyError as error:
      raise ValidationError(f"Unsupported Whisper model: {model_id}") from error

  def is_cached(self, model_id: str) -> bool:
    """Return whether complete Whisper weights are cached locally."""
    return is_repository_cached(self._cache_root, self._repository(model_id))

  def cache_size(self, model_id: str) -> int | None:
    """Return cached Whisper bytes."""
    return repository_size(self._cache_root, self._repository(model_id))

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Download Whisper weights without loading them and report byte progress."""
    await download_repository(self._cache_root, self._repository(model_id), progress)

  async def load(self, model_id: str) -> None:
    """Download if necessary and load the selected Whisper model."""
    async with self._lock:
      if self._loaded_model == model_id and self._model is not None:
        return
      if self._model is not None:
        await self._unload_locked()
      if not await asyncio.to_thread(self.is_cached, model_id):
        await self.download(model_id)
      (
        processor,
        model,
        dtype,
        parameters,
        accepts_kwargs,
      ) = await run_in_daemon_thread(
        self._load_sync,
        model_id,
        thread_name="voice-whisper-load",
      )
      self._processor = processor
      self._model = model
      self._model_dtype = dtype
      self._generate_parameters = parameters
      self._generate_accepts_kwargs = accepts_kwargs
      self._loaded_model = model_id

  def _load_sync(
    self,
    model_id: str,
  ) -> tuple[Any, Any, Any, set[str], bool]:
    """Load processor and model through blocking Transformers APIs."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    repository = repository_snapshot_path(
      self._cache_root,
      self._repository(model_id),
    )
    processor = WhisperProcessor.from_pretrained(
      repository,
      local_files_only=True,
    )
    dtype = self._torch.bfloat16 if self._device == "cuda" else self._torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(
      repository,
      local_files_only=True,
      dtype=dtype,
    )
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
      generation_config.forced_decoder_ids = None
    model_config = getattr(model, "config", None)
    if model_config is not None:
      model_config.forced_decoder_ids = None
    cast(Any, model).to(self._device)
    if self._device == "cuda":
      try:
        capability = self._torch.cuda.get_device_capability()
        mode = "max-autotune" if capability[0] >= 9 else "reduce-overhead"
        model = cast(Any, self._torch.compile(model, mode=mode))
      except Exception as error:
        LOGGER.warning("Whisper torch.compile was skipped: %s", error)
    signature = inspect.signature(model.generate)
    parameters = set(signature.parameters)
    accepts_kwargs = any(
      parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )
    return processor, model, dtype, parameters, accepts_kwargs

  async def unload(self) -> None:
    """Release Whisper and clear accelerator caches."""
    async with self._lock:
      await self._unload_locked()

  async def _unload_locked(self) -> None:
    """Release Whisper while the lifecycle lock is held."""
    self._model = None
    self._processor = None
    self._loaded_model = None
    self._generate_parameters.clear()
    self._generate_accepts_kwargs = False
    if self._torch.cuda.is_available():
      self._torch.cuda.empty_cache()

  async def delete(self, model_id: str) -> None:
    """Unload the selected Whisper model and remove its cache."""
    if self._loaded_model == model_id:
      await self.unload()
    await delete_repository(self._cache_root, self._repository(model_id))

  async def transcribe(
    self,
    request: TranscriptionRequest,
  ) -> TranscriptionResult:
    """Transcribe one audio file outside the event loop."""
    if self._model is None or self._processor is None:
      raise RuntimeError("Whisper model is not loaded.")
    text = await run_in_daemon_thread(
      self._transcribe_sync,
      request.audio_path,
      request.language,
      request.context,
      thread_name="voice-whisper-transcribe",
    )
    return TranscriptionResult(text=text, language=request.language)

  def _transcribe_sync(
    self,
    audio_path: Path,
    language: str | None,
    initial_prompt: str | None,
  ) -> str:
    """Run blocking feature extraction, generation, and decoding."""
    model = self._model
    processor = self._processor
    if model is None or processor is None:
      raise RuntimeError("Whisper model is not loaded.")
    audio = load_audio(audio_path, sample_rate=16_000)
    long_form = len(audio.samples) > 30 * audio.sample_rate
    inputs = processor(
      audio.samples,
      sampling_rate=16_000,
      # Preserve the entire recording; Whisper otherwise truncates at 30 seconds.
      truncation=False,
      padding="longest" if long_form else "max_length",
      return_attention_mask=True,
      return_tensors="pt",
    )
    features = inputs["input_features"].to(self._device)
    attention_mask = inputs["attention_mask"].to(self._device)
    if features.is_floating_point():
      features = features.to(dtype=self._model_dtype)
    kwargs: dict[str, Any] = {"attention_mask": attention_mask}
    if long_form:
      kwargs["return_timestamps"] = True
    if self._supports_generate_argument("task"):
      kwargs["task"] = "transcribe"
    if language:
      if self._supports_generate_argument("language"):
        kwargs["language"] = language
      else:
        kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
          language=language,
          task="transcribe",
        )
    if initial_prompt and self._supports_generate_argument("prompt_ids"):
      prompt_ids = self._prompt_ids(initial_prompt)
      if prompt_ids is not None:
        kwargs["prompt_ids"] = prompt_ids
    with self._torch.inference_mode():
      predicted = model.generate(features, **kwargs)
    decoded = processor.batch_decode(predicted, skip_special_tokens=True)
    return str(decoded[0]).strip()

  def _supports_generate_argument(self, name: str) -> bool:
    """Return whether the loaded generate method accepts an argument."""
    return name in self._generate_parameters or self._generate_accepts_kwargs

  def _prompt_ids(self, prompt: str) -> Any | None:
    """Convert a text prompt into device-local IDs when supported."""
    processor = self._processor
    if processor is None or not hasattr(processor, "get_prompt_ids"):
      return None
    try:
      prompt_ids = processor.get_prompt_ids(prompt, return_tensors="pt")
    except TypeError:
      prompt_ids = processor.get_prompt_ids(prompt)
    if prompt_ids is None:
      return None
    if isinstance(prompt_ids, self._torch.Tensor):
      return prompt_ids.to(self._device)
    return self._torch.tensor(np.asarray(prompt_ids), device=self._device)
