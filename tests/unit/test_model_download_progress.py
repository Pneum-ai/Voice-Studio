"""Model lifecycle byte-progress regressions."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from app.core.config import Settings
from app.domain.jobs import JobState
from app.domain.models import DesiredModelState
from app.services.jobs import JobService
from app.services.models import ModelService
from tests.helpers import fake_backends


@pytest.mark.asyncio
async def test_download_bytes_update_job_progress_and_detail() -> None:
  """Map a backend byte callback into a visible proportional job snapshot."""
  backends = fake_backends()
  backend = cast(Any, backends.tts)
  backend.cached.remove("qwen-tts-0.6B")
  halfway = asyncio.Event()
  finish = asyncio.Event()

  async def download(model_id: str, progress: Any = None) -> None:
    """Pause a deterministic fake download after its first byte update."""
    assert model_id == "qwen-tts-0.6B"
    await progress(25, 100)
    halfway.set()
    await finish.wait()
    await progress(100, 100)
    backend.cached.add(model_id)

  backend.download = download
  jobs = JobService(300)
  models = ModelService(Settings(prewarm=False), backends, jobs)

  job = await models.request_state("qwen-tts-0.6B", DesiredModelState.DOWNLOADED)
  await halfway.wait()
  active = await jobs.get(job.id)

  assert active.state is JobState.RUNNING
  assert active.progress == pytest.approx(0.275)
  assert active.stage == "downloading"
  assert active.detail.startswith("25 B of 100 B")

  finish.set()
  while not (completed := await jobs.get(job.id)).terminal:
    await asyncio.sleep(0)

  assert completed.state is JobState.COMPLETE
  assert completed.detail == "Finished successfully"
  assert backend.is_cached("qwen-tts-0.6B")
