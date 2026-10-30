"""MLX transcription for Whisper, Qwen3-ASR, Canary, and Parakeet."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from app.backends.cache import (
  cache_root,
  delete_repository,
  download_repository,
  is_repository_cached,
  repository_size,
  repository_snapshot_path,
)
from app.backends.mlx.worker import MLXWorker
from app.backends.protocols import (
  DownloadProgressCallback,
  TranscriptionRequest,
  TranscriptionResult,
)
from app.backends.pytorch.qwen_asr_worker import LANGUAGE_NAMES
from app.core.errors import ValidationError

MODEL_REPOSITORIES = {
  "nvidia-canary-1b-v2": "Mediform/canary-1b-v2-mlx-q8",
  "nvidia-parakeet-tdt-0.6b-v3": "mlx-community/parakeet-tdt-0.6b-v3",
  "qwen3-asr-0.6B": "mlx-community/Qwen3-ASR-0.6B-8bit",
  "qwen3-asr-1.7B": "mlx-community/Qwen3-ASR-1.7B-8bit",
  "whisper-base": "mlx-community/whisper-base-mlx",
  "whisper-small": "mlx-community/whisper-small-mlx",
  "whisper-medium": "mlx-community/whisper-medium-mlx",
  "whisper-large": "mlx-community/whisper-large-v3-mlx",
  "whisper-turbo": "mlx-community/whisper-large-v3-turbo-asr-fp16",
}


class MLXSTTBackend:
  """Run the supported transcription families through mlx-audio."""

  def __init__(
    self, model_cache_dir: Path | None = None, *, worker: MLXWorker | None = None
  ) -> None:
    """Initialize an unloaded MLX speech recognition backend."""
    self._worker = worker or MLXWorker()
    self._cache_root = cache_root(model_cache_dir)
    self._model: Any | None = None
    self._loaded_model: str | None = None
    self._lock = asyncio.Lock()

  @property
  def loaded_model(self) -> str | None:
    """Return the canonical speech recognition model ID in memory."""
    return self._loaded_model

  def supports(self, model_id: str) -> bool:
    """Return whether this backend supports the speech recognition model."""
    return model_id in MODEL_REPOSITORIES

  def _repository(self, model_id: str) -> str:
    """Resolve a canonical model ID to its repository."""
    try:
      return MODEL_REPOSITORIES[model_id]
    except KeyError as error:
      raise ValidationError(f"Unsupported speech recognition model: {model_id}") from error

  def is_cached(self, model_id: str) -> bool:
    """Return whether complete speech recognition weights are cached."""
    return is_repository_cached(self._cache_root, self._repository(model_id))

  def cache_size(self, model_id: str) -> int | None:
    """Return cached speech recognition bytes."""
    return repository_size(self._cache_root, self._repository(model_id))

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Download speech recognition weights without loading them and report byte progress."""
    await download_repository(self._cache_root, self._repository(model_id), progress)

  async def load(self, model_id: str) -> None:
    """Download if necessary and load the selected speech recognition model."""
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

  def _load_sync(self, model_id: str) -> Any:
    """Load a converted checkpoint with an explicit architecture."""
    from mlx_audio.stt.utils import load_model  # pyright: ignore[reportMissingImports]

    snapshot = repository_snapshot_path(
      self._cache_root,
      self._repository(model_id),
    )
    # Cache snapshots have hash names; older converted configs omit model_type.
    model_type = "whisper"
    if model_id.startswith("qwen3-asr"):
      model_type = "qwen3_asr"
    elif model_id.startswith("nvidia-canary"):
      model_type = "canary"
    elif model_id.startswith("nvidia-parakeet"):
      model_type = "parakeet"
    return load_model(snapshot, model_type=model_type)

  async def unload(self) -> None:
    """Release the loaded speech recognition model."""
    async with self._lock:
      await self._unload_locked()

  async def _unload_locked(self) -> None:
    """Release the loaded model while the lifecycle lock is held."""
    await self._worker.run(self._release_sync)
    self._loaded_model = None

  def _release_sync(self) -> None:
    """Drop model state on the thread that owns its GPU streams."""
    self._model = None

  async def delete(self, model_id: str) -> None:
    """Unload the selected model and delete its weights."""
    if self._loaded_model == model_id:
      await self.unload()
    await delete_repository(self._cache_root, self._repository(model_id))

  async def transcribe(
    self,
    request: TranscriptionRequest,
  ) -> TranscriptionResult:
    """Transcribe audio outside the event loop."""
    model = self._model
    if model is None:
      raise RuntimeError("MLX speech recognition model is not loaded.")

    def transcribe_sync() -> str:
      """Run blocking MLX transcription and normalize its result shape."""
      options = self._transcription_options(request)
      result = model.generate(str(request.audio_path), **options)
      if isinstance(result, str):
        return result.strip()
      if isinstance(result, dict):
        return str(result.get("text", "")).strip()
      if hasattr(result, "text"):
        return str(result.text).strip()
      return str(result).strip()

    text = await self._worker.run(
      transcribe_sync,
    )
    return TranscriptionResult(text=text, language=request.language)

  def _transcription_options(self, request: TranscriptionRequest) -> dict[str, Any]:
    """Translate API language codes and context to each MLX model's native options."""
    model_id = self._loaded_model or ""
    if model_id.startswith("nvidia-canary"):
      if request.context:
        raise ValidationError(
          "Canary on MLX does not support recognition context.", "unsupported_context"
        )
      if not request.language:
        raise ValidationError("Canary requires a language.", "language_required")
      return {
        "source_lang": request.language,
        "target_lang": request.language,
        "max_tokens": 8192,
      }
    if model_id.startswith("nvidia-parakeet"):
      return {"chunk_duration": 30.0}
    options: dict[str, Any] = {}
    if model_id.startswith("qwen3-asr"):
      if request.language:
        options["language"] = LANGUAGE_NAMES[request.language]
      if request.context:
        options["system_prompt"] = request.context
    else:
      if request.language:
        options["language"] = request.language
      if request.context:
        options["initial_prompt"] = request.context
    return options
