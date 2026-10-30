"""Model catalog, lifecycle jobs, prewarming, and readiness checks."""

from __future__ import annotations

import asyncio
import gc
import logging
import os
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import replace

from app.backends.factory import BackendBundle
from app.backends.protocols import ModelBackend
from app.core.config import Settings
from app.core.errors import (
  ConflictError,
  InferenceUnavailableError,
  ModelNotReadyError,
  ValidationError,
)
from app.domain.jobs import Job, JobState
from app.domain.model_catalog import MODEL_DEFINITIONS
from app.domain.models import DesiredModelState, ModelDefinition, ModelKind, ModelStatus
from app.services.jobs import JobService
from app.services.model_selection import MODEL_SWITCH_RESTART_EXIT_CODE, ModelSelectionStore
from app.utils.concurrency import SharedExclusiveGuard

LOGGER = logging.getLogger(__name__)
DOWNLOAD_PROGRESS_START = 0.05


def _format_bytes(size: float) -> str:
  """Format a byte count for compact lifecycle-job detail."""
  value = max(0.0, size)
  units = ("B", "KB", "MB", "GB", "TB")
  unit = units[0]
  for candidate in units:
    unit = candidate
    if value < 1024 or candidate == units[-1]:
      break
    value /= 1024
  precision = 0 if value >= 10 or unit == "B" else 1
  return f"{value:.{precision}f} {unit}"


def _format_duration(seconds: float) -> str:
  """Format a nonnegative ETA in seconds or minutes and seconds."""
  rounded = max(0, round(seconds))
  minutes, remaining = divmod(rounded, 60)
  return f"{minutes}m {remaining}s" if minutes else f"{remaining}s"


class ModelService:
  """Expose a stable catalog over backend-specific lifecycle implementations."""

  def __init__(
    self,
    settings: Settings,
    backends: BackendBundle,
    jobs: JobService,
    *,
    model_selection: ModelSelectionStore | None = None,
    restart_process: Callable[[int], None] = os._exit,
    restart_delay_seconds: float = 0.5,
  ) -> None:
    """Store backend dependencies and serialize lifecycle work by model kind."""
    self.settings = settings
    self.backends = backends
    self.jobs = jobs
    self._model_selection = model_selection or ModelSelectionStore(settings.model_selection_path)
    self._restart_process = restart_process
    self._restart_delay_seconds = restart_delay_seconds
    self._definitions = {definition.id: definition for definition in MODEL_DEFINITIONS}
    if backends.name == "mlx":
      canary = self._definitions["nvidia-canary-1b-v2"]
      if canary.transcription is not None:
        self._definitions[canary.id] = replace(
          canary, transcription=replace(canary.transcription, context=False)
        )
    self._guards = {
      ModelKind.TTS: SharedExclusiveGuard(),
      ModelKind.TRANSCRIPTION: SharedExclusiveGuard(),
    }

  def definition(self, model_id: str) -> ModelDefinition:
    """Return a catalog definition or raise a request validation error."""
    try:
      return self._definitions[model_id]
    except KeyError as error:
      raise ValidationError(f"Unknown model: {model_id}", "unknown_model") from error

  def backend_for(self, model_id: str) -> ModelBackend:
    """Return the backend responsible for a catalog model."""
    definition = self.definition(model_id)
    if definition.kind is ModelKind.TTS:
      return self.backends.tts
    return self.backends.stt

  async def list_models(self) -> list[ModelStatus]:
    """Return status for every model in deterministic display order."""
    return list(await asyncio.gather(*(self.get_model(item.id) for item in MODEL_DEFINITIONS)))

  async def get_model(self, model_id: str) -> ModelStatus:
    """Return backend capability and local state for one model."""
    definition = self.definition(model_id)
    backend = self.backend_for(model_id)
    supported = backend.supports(model_id)
    active_job = await self.jobs.active_job_for(model_id)
    runtime_status = getattr(backend, "runtime_status", None)
    runtime_ready = runtime_status is None or runtime_status.ready
    native_streaming = (
      definition.kind is ModelKind.TTS and supported and self.backends.tts.supports_streaming
    )
    downloaded = False
    size_bytes = None
    if supported:
      downloaded, size_bytes = await asyncio.gather(
        asyncio.to_thread(backend.is_cached, model_id),
        asyncio.to_thread(backend.cache_size, model_id),
      )
    return ModelStatus(
      id=definition.id,
      display_name=definition.display_name,
      kind=definition.kind,
      family=definition.family,
      provider=definition.provider,
      transcription=definition.transcription,
      speech=definition.speech,
      downloaded=downloaded,
      loaded=supported and backend.loaded_model == model_id and runtime_ready,
      size_bytes=size_bytes,
      backend_supported=supported,
      native_streaming=native_streaming,
      active_job_id=active_job.id if active_job else None,
    )

  async def request_state(
    self,
    model_id: str,
    desired_state: DesiredModelState,
  ) -> Job:
    """Schedule a model download, load, unload, or deletion transition."""
    definition = self.definition(model_id)
    backend = self.backend_for(model_id)
    if not backend.supports(model_id):
      raise ValidationError(
        f"{model_id} is not supported by the {self.backends.name} backend.",
        "unsupported_model",
      )
    if await self.jobs.active_job_for(model_id):
      raise ConflictError(f"A lifecycle job is already active for {model_id}.")

    async def transition(job_id: str) -> None:
      """Apply one desired state while serializing shared backend memory."""
      async with self._guards[definition.kind].exclusive():
        if desired_state is DesiredModelState.ABSENT:
          await self.jobs.update(job_id, progress=0.2, stage="unloading")
          if backend.loaded_model == model_id:
            await backend.unload()
          await self.jobs.update(job_id, progress=0.5, stage="deleting")
          await backend.delete(model_id)
        elif desired_state is DesiredModelState.DOWNLOADED:
          download_start = DOWNLOAD_PROGRESS_START
          if backend.loaded_model == model_id:
            await self.jobs.update(
              job_id,
              progress=0.1,
              stage="unloading",
              detail="Releasing the model from memory",
            )
            await backend.unload()
            download_start = 0.2
          if not await asyncio.to_thread(backend.is_cached, model_id):
            await self._download_model(
              job_id,
              backend,
              model_id,
              start_progress=download_start,
              end_progress=0.95,
            )
        else:
          if not await asyncio.to_thread(backend.is_cached, model_id):
            await self._download_model(
              job_id,
              backend,
              model_id,
              start_progress=DOWNLOAD_PROGRESS_START,
              end_progress=0.65,
            )
          await self.jobs.update(
            job_id,
            progress=0.7,
            stage="loading",
            detail="Loading model weights into memory",
          )
          if definition.kind is ModelKind.TTS:
            await self._load_tts(job_id, model_id)
          else:
            await backend.load(model_id)

    return await self.jobs.start(
      kind="model_state",
      resource_id=model_id,
      work=transition,
    )

  async def _load_tts(self, job_id: str, model_id: str) -> None:
    """Load TTS with a clean CUDA lifecycle when replacing another model."""
    loaded = self.backends.tts.loaded_model
    if loaded is not None and loaded != model_id:
      async with self._guards[ModelKind.TRANSCRIPTION].exclusive():
        if self.settings.supervisor_enabled and self.settings.prewarm:
          await self._request_supervised_restart(job_id, model_id)
        else:
          await self._replace_tts_in_process(job_id, model_id)
    else:
      await self.backends.tts.load(model_id)
    await asyncio.to_thread(self._model_selection.select, model_id)

  async def _request_supervised_restart(self, job_id: str, model_id: str) -> None:
    """Persist job state and terminate so the supervisor resets CUDA fully."""
    job = await self.jobs.update(
      job_id,
      progress=0.72,
      stage="restarting",
      detail="Restarting the CUDA worker to release model memory",
    )
    await asyncio.to_thread(self._model_selection.prepare, job, model_id)
    LOGGER.warning(
      "Restarting the supervised CUDA worker to switch TTS from %s to %s.",
      self.backends.tts.loaded_model,
      model_id,
    )
    await asyncio.sleep(self._restart_delay_seconds)
    self._restart_process(MODEL_SWITCH_RESTART_EXIT_CODE)
    raise RuntimeError("The configured supervisor restart callback returned unexpectedly.")

  async def _replace_tts_in_process(self, job_id: str, model_id: str) -> None:
    """Mirror clean startup order when no process supervisor is available."""
    transcription_model = self.backends.stt.loaded_model
    await self.jobs.update(
      job_id,
      progress=0.72,
      stage="releasing",
      detail="Releasing speech, transcription, and CUDA graph memory",
    )
    if transcription_model is not None:
      await self.backends.stt.unload()
    await self.backends.tts.unload()
    await asyncio.to_thread(gc.collect)
    try:
      await self.backends.tts.load(model_id)
    finally:
      if transcription_model is not None:
        await self.backends.stt.load(transcription_model)

  async def _download_model(
    self,
    job_id: str,
    backend: ModelBackend,
    model_id: str,
    *,
    start_progress: float,
    end_progress: float,
  ) -> None:
    """Map backend byte counts into one bounded lifecycle-job phase."""
    started_at = time.monotonic()
    initial_bytes: int | None = None
    await self.jobs.update(
      job_id,
      progress=start_progress,
      stage="downloading",
      detail="Reading remote model size",
    )

    async def report(current_bytes: int, total_bytes: int | None) -> None:
      """Publish byte count, rate, ETA, and proportional job progress."""
      nonlocal initial_bytes
      if initial_bytes is None:
        initial_bytes = current_bytes
      elapsed = max(time.monotonic() - started_at, 0.001)
      transferred = max(0, current_bytes - initial_bytes)
      rate = transferred / elapsed
      if total_bytes:
        fraction = max(0.0, min(current_bytes / total_bytes, 1.0))
        progress = start_progress + (end_progress - start_progress) * fraction
        detail = f"{_format_bytes(current_bytes)} of {_format_bytes(total_bytes)}"
        if rate > 0:
          remaining = max(0, total_bytes - current_bytes) / rate
          detail += f" · {_format_bytes(rate)}/s · {_format_duration(remaining)} remaining"
      else:
        progress = start_progress
        detail = f"{_format_bytes(current_bytes)} downloaded"
        if rate > 0:
          detail += f" · {_format_bytes(rate)}/s"
      await self.jobs.update(
        job_id,
        progress=progress,
        stage="downloading",
        detail=detail,
      )

    await backend.download(model_id, report)
    await self.jobs.update(
      job_id,
      progress=end_progress,
      stage="downloaded",
      detail="Model download complete",
    )

  def require_loaded(self, model_id: str, kind: ModelKind) -> None:
    """Require a supported model of the expected kind to be in memory."""
    definition = self.definition(model_id)
    if definition.kind is not kind:
      raise ValidationError(
        f"{model_id} is not a {kind.value} model.",
        "wrong_model_kind",
      )
    backend = self.backend_for(model_id)
    if not backend.supports(model_id) or backend.loaded_model != model_id:
      raise ModelNotReadyError(model_id)
    runtime_status = getattr(backend, "runtime_status", None)
    if runtime_status is not None and not runtime_status.ready:
      raise InferenceUnavailableError(
        runtime_status.reason,
        runtime_status.failure_id,
      )

  @asynccontextmanager
  async def use_loaded(
    self,
    model_id: str,
    kind: ModelKind,
  ) -> AsyncIterator[None]:
    """Hold shared lifecycle access while a loaded model is in active use.

    Access is shared so concurrent requests can overlap and be batched. Only
    lifecycle transitions need exclusive access, so holding this for the whole
    of a generation no longer serializes inference.
    """
    async with self._guards[kind].shared():
      self.require_loaded(model_id, kind)
      yield

  async def prewarm_defaults(self) -> None:
    """Download when needed and load both configured default models."""
    if not self.settings.prewarm:
      LOGGER.info("Default model prewarming is disabled.")
      return
    selection = await asyncio.to_thread(self._model_selection.read)
    speech_model = selection.selected_model or self.settings.default_qwen_model
    if not self.backends.tts.supports(speech_model):
      LOGGER.error("Ignoring unsupported persisted TTS selection %s.", speech_model)
      speech_model = self.settings.default_qwen_model
      await asyncio.to_thread(self._model_selection.select, speech_model)
    for model_id in (
      speech_model,
      self.settings.default_transcription_model,
    ):
      backend = self.backend_for(model_id)
      if not backend.supports(model_id):
        raise RuntimeError(
          f"The configured default model {model_id} is unsupported by "
          f"the {self.backends.name} backend."
        )
      cached = await asyncio.to_thread(backend.is_cached, model_id)
      LOGGER.info(
        "Prewarming default model %s (cached=%s).",
        model_id,
        cached,
      )
      await backend.load(model_id)
      LOGGER.info("Default model %s is loaded.", model_id)
    LOGGER.info(
      "Default model prewarming complete: speech=%s | transcription=%s.",
      speech_model,
      self.settings.default_transcription_model,
    )
    pending = selection.pending
    if pending is not None:
      if pending.model_id == speech_model:
        await self.jobs.update(
          pending.job_id,
          state=JobState.COMPLETE,
          progress=1.0,
          stage="complete",
          detail="Model loaded after a clean CUDA worker restart",
        )
      else:
        await self.jobs.update(
          pending.job_id,
          state=JobState.FAILED,
          stage="failed",
          detail="The persisted model selection was invalid",
          error=f"Unsupported persisted TTS model: {pending.model_id}",
        )
      await asyncio.to_thread(self._model_selection.complete, pending.job_id)

  async def unload_all(self) -> None:
    """Release all loaded inference models during application shutdown."""
    await asyncio.gather(
      self.backends.tts.unload(),
      self.backends.stt.unload(),
    )
