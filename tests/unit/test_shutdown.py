"""Cancellation-friendly process shutdown behavior."""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.domain.jobs import JobState
from app.services.jobs import JobService
from app.utils.async_work import run_in_daemon_thread


@pytest.mark.asyncio
async def test_job_shutdown_cancels_active_work_and_rejects_new_jobs() -> None:
  """Cancel active lifecycle jobs before model cleanup starts."""
  service = JobService(300)
  started = asyncio.Event()
  cancelled = asyncio.Event()

  async def wait_forever(_: str) -> None:
    """Remain active until shutdown cancels the job."""
    started.set()
    try:
      await asyncio.Event().wait()
    finally:
      cancelled.set()

  job = await service.start(
    kind="model_state",
    resource_id="qwen-tts-1.7B",
    work=wait_forever,
  )
  await started.wait()

  await service.shutdown()

  assert cancelled.is_set()
  assert (await service.get(job.id)).state is JobState.CANCELLED
  with pytest.raises(RuntimeError, match="shutting down"):
    await service.start(
      kind="model_state",
      resource_id="whisper-turbo",
      work=wait_forever,
    )


@pytest.mark.asyncio
async def test_cancelled_inference_thread_cannot_hold_process_shutdown() -> None:
  """Use daemon workers so cancelled native inference cannot block exit."""
  release = threading.Event()
  started = threading.Event()
  daemon_state: list[bool] = []

  def blocking_inference() -> None:
    """Model a native operation that does not respond to task cancellation."""
    daemon_state.append(threading.current_thread().daemon)
    started.set()
    release.wait(timeout=5)

  task = asyncio.create_task(
    run_in_daemon_thread(
      blocking_inference,
      thread_name="test-inference",
    )
  )
  while not started.is_set():
    await asyncio.sleep(0)
  task.cancel()

  with pytest.raises(asyncio.CancelledError):
    await asyncio.wait_for(task, timeout=0.5)
  assert daemon_state == [True]
  release.set()
