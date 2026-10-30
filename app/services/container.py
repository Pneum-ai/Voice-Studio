"""Explicit application service graph and lifecycle management."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.backends.factory import BackendBundle, create_backends
from app.backends.pytorch.nari_streaming import NariRuntimeConfig
from app.core.config import Settings
from app.persistence.database import Database
from app.persistence.generations import GenerationRepository
from app.persistence.profiles import ProfileRepository
from app.services.generations import GenerationService
from app.services.jobs import JobService
from app.services.model_runtimes import ModelRuntimeService
from app.services.model_selection import ModelSelectionStore
from app.services.models import ModelService
from app.services.profiles import ProfileService
from app.services.prompts import PromptService
from app.services.speech import SpeechService
from app.services.transcriptions import TranscriptionService
from app.services.tts_batcher import TTSBatcher

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Services:
  """Hold all process-scoped dependencies without module-level mutable state."""

  settings: Settings
  database: Database
  backends: BackendBundle
  jobs: JobService
  models: ModelService
  model_runtimes: ModelRuntimeService
  profiles: ProfileService
  prompts: PromptService
  generations: GenerationService
  speech: SpeechService
  transcriptions: TranscriptionService
  tts_batcher: TTSBatcher

  async def startup(self) -> None:
    """Initialize storage, verify integrity, and prewarm default models."""
    self.settings.create_runtime_directories()
    await self.database.initialize()
    await self.database.check_integrity()
    await self.model_runtimes.ensure_all()
    await self.models.prewarm_defaults()

  async def shutdown(self) -> None:
    """Release model memory and close database connections."""
    LOGGER.info("Shutdown phase 1/3: cancelling active model jobs.")
    await self.jobs.shutdown()
    await self.tts_batcher.stop()
    LOGGER.info("Shutdown phase 2/3: unloading speech and transcription models.")
    try:
      await self.models.unload_all()
      LOGGER.info("Speech and transcription models are unloaded.")
    finally:
      LOGGER.info("Shutdown phase 3/3: closing SQLite connections.")
      await self.database.close()
      LOGGER.info("Application resources closed cleanly.")


def build_services(
  settings: Settings,
  backend_bundle: BackendBundle | None = None,
) -> Services:
  """Construct the complete dependency graph for one application instance."""
  backends = backend_bundle or create_backends(
    settings.model_cache_dir,
    qwen_asr_python=settings.qwen_asr_python,
    runtime_dir=settings.runtime_dir,
    torch_compile=settings.torch_compile,
    qwen_engine=settings.qwen_engine,
    nari_config=NariRuntimeConfig(
      execution_profile=settings.nari_execution_profile,
      policy_mode=settings.nari_policy_mode,
      max_active_0_6b=settings.nari_max_active_0_6b,
      max_active_1_7b=settings.nari_max_active_1_7b,
      blackwell_fp8=settings.nari_blackwell_fp8,
      capture_max_batch=settings.nari_capture_max_batch,
      max_queued=settings.nari_max_queued,
      queue_timeout_ms=settings.nari_queue_timeout_ms,
      first_audio_timeout_seconds=settings.nari_first_audio_timeout_seconds,
      request_timeout_seconds=settings.nari_request_timeout_seconds,
      silence_trim=settings.nari_silence_trim,
      silence_threshold_dbfs=settings.nari_silence_threshold_dbfs,
      silence_max_ms=settings.nari_silence_max_ms,
      silence_preroll_ms=settings.nari_silence_preroll_ms,
      silence_min_output_ms=settings.nari_silence_min_output_ms,
      chunk_schedule=settings.nari_chunk_schedule,
      low_latency_chunk_schedule=settings.nari_low_latency_chunk_schedule,
      low_latency_max_active=settings.nari_low_latency_max_active,
      adaptive_target_ms=settings.nari_adaptive_target_ms,
      watchdog_stall_seconds=settings.nari_watchdog_stall_seconds,
      exit_on_failure=settings.nari_exit_on_failure,
    ),
  )
  database = Database(settings.database_path)
  profile_repository = ProfileRepository()
  generation_repository = GenerationRepository()
  model_selection = ModelSelectionStore(settings.model_selection_path)
  pending_switch = model_selection.read().pending
  restored_jobs = (pending_switch.as_job(),) if pending_switch is not None else ()
  jobs = JobService(settings.job_retention_seconds, restored_jobs)
  models = ModelService(settings, backends, jobs, model_selection=model_selection)
  model_runtimes = ModelRuntimeService(
    settings,
    jobs,
    auto_install=(
      backend_bundle is None and backends.name != "mlx" and settings.auto_install_runtimes
    ),
  )
  prompts = PromptService(settings, profile_repository, backends)
  profiles = ProfileService(
    settings,
    profile_repository,
    cache_invalidator=prompts.clear_profile,
  )
  generations = GenerationService(settings, generation_repository)
  tts_batcher = TTSBatcher(
    backends,
    enabled=settings.batch_enabled,
    max_batch_size=settings.batch_max_size,
    window_seconds=settings.batch_window_ms / 1000,
  )
  speech = SpeechService(
    backends,
    models,
    profiles,
    prompts,
    tts_batcher,
  )
  transcriptions = TranscriptionService(backends, models)
  return Services(
    settings=settings,
    database=database,
    backends=backends,
    jobs=jobs,
    models=models,
    model_runtimes=model_runtimes,
    profiles=profiles,
    prompts=prompts,
    generations=generations,
    speech=speech,
    transcriptions=transcriptions,
    tts_batcher=tts_batcher,
  )
