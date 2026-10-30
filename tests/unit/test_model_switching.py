"""Restart-safe TTS model selection and fallback lifecycle tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from app.core.config import Settings
from app.domain.jobs import Job, JobState
from app.domain.models import DesiredModelState
from app.services.jobs import JobService
from app.services.model_selection import ModelSelectionStore
from app.services.models import MODEL_SWITCH_RESTART_EXIT_CODE, ModelService
from tests.helpers import fake_backends


async def _wait_for_terminal(jobs: JobService, job_id: str) -> Job:
  """Wait briefly for deterministic fake lifecycle work to finish."""
  for _ in range(100):
    job = await jobs.get(job_id)
    if job.terminal:
      return job
    await asyncio.sleep(0.01)
  raise AssertionError("model lifecycle job did not finish")


def _pending_job() -> Job:
  """Build a stable job snapshot for restart restoration."""
  timestamp = datetime.now(UTC)
  return Job(
    id="switch-job",
    kind="model_state",
    resource_id="qwen-tts-1.7B",
    state=JobState.RUNNING,
    progress=0.72,
    stage="restarting",
    detail="Restarting",
    error=None,
    created_at=timestamp,
    updated_at=timestamp,
  )


@pytest.mark.asyncio
async def test_supervised_switch_persists_job_before_cuda_restart(tmp_path: Path) -> None:
  """Persist the target and original API job before requesting worker exit."""
  selection = ModelSelectionStore(tmp_path / "selection.json")
  jobs = JobService(300)
  restart_codes: list[int] = []
  service = ModelService(
    Settings(
      prewarm=True,
      supervisor_enabled=True,
      model_selection_file=selection.path,
    ),
    fake_backends(),
    jobs,
    model_selection=selection,
    restart_process=restart_codes.append,
    restart_delay_seconds=0,
  )

  job = await service.request_state("qwen-tts-0.6B", DesiredModelState.LOADED)
  terminal = await _wait_for_terminal(jobs, job.id)

  state = selection.read()
  assert restart_codes == [MODEL_SWITCH_RESTART_EXIT_CODE]
  assert terminal.state is JobState.FAILED
  assert state.selected_model == "qwen-tts-0.6B"
  assert state.pending is not None and state.pending.job_id == job.id


@pytest.mark.asyncio
async def test_restart_loads_selected_tts_before_stt_and_completes_job(
  tmp_path: Path,
) -> None:
  """Restore a pending job and finish it after clean-order model prewarming."""
  selection = ModelSelectionStore(tmp_path / "selection.json")
  pending = _pending_job()
  selection.prepare(pending, "qwen-tts-1.7B")
  backends = fake_backends()
  tts = cast(Any, backends.tts)
  stt = cast(Any, backends.stt)
  tts._loaded_model = None
  stt._loaded_model = None
  order: list[str] = []
  original_tts_load = tts.load
  original_stt_load = stt.load

  async def load_tts(model_id: str) -> None:
    """Record fake TTS startup ordering."""
    order.append(f"tts:{model_id}")
    await original_tts_load(model_id)

  async def load_stt(model_id: str) -> None:
    """Record fake STT startup ordering."""
    order.append(f"stt:{model_id}")
    await original_stt_load(model_id)

  tts.load = load_tts
  stt.load = load_stt
  restored = selection.read().pending
  assert restored is not None
  jobs = JobService(300, (restored.as_job(),))
  service = ModelService(
    Settings(prewarm=True, model_selection_file=selection.path),
    backends,
    jobs,
    model_selection=selection,
  )

  await service.prewarm_defaults()

  assert order == ["tts:qwen-tts-1.7B", "stt:whisper-turbo"]
  assert (await jobs.get(pending.id)).state is JobState.COMPLETE
  assert selection.read().pending is None
  assert selection.read().selected_model == "qwen-tts-1.7B"


@pytest.mark.asyncio
async def test_unsupervised_switch_releases_stt_and_restores_clean_order(
  tmp_path: Path,
) -> None:
  """Avoid fragmented reloads even when the server was launched directly."""
  backends = fake_backends()
  tts = cast(Any, backends.tts)
  stt = cast(Any, backends.stt)
  order: list[str] = []
  original_tts_unload = tts.unload
  original_stt_unload = stt.unload
  original_tts_load = tts.load
  original_stt_load = stt.load

  async def unload_tts() -> None:
    """Record TTS release."""
    order.append("tts:unload")
    await original_tts_unload()

  async def unload_stt() -> None:
    """Record STT release."""
    order.append("stt:unload")
    await original_stt_unload()

  async def load_tts(model_id: str) -> None:
    """Record TTS load."""
    order.append(f"tts:{model_id}")
    await original_tts_load(model_id)

  async def load_stt(model_id: str) -> None:
    """Record STT restoration."""
    order.append(f"stt:{model_id}")
    await original_stt_load(model_id)

  tts.unload = unload_tts
  stt.unload = unload_stt
  tts.load = load_tts
  stt.load = load_stt
  selection = ModelSelectionStore(tmp_path / "selection.json")
  jobs = JobService(300)
  service = ModelService(
    Settings(prewarm=True, model_selection_file=selection.path),
    backends,
    jobs,
    model_selection=selection,
  )

  job = await service.request_state("qwen-tts-0.6B", DesiredModelState.LOADED)
  terminal = await _wait_for_terminal(jobs, job.id)

  assert terminal.state is JobState.COMPLETE
  assert order == [
    "stt:unload",
    "tts:unload",
    "tts:qwen-tts-0.6B",
    "stt:whisper-turbo",
  ]
  assert selection.read().selected_model == "qwen-tts-0.6B"
