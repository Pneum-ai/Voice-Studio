"""In-memory job tracking with retained snapshots and SSE subscriptions."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from app.core.errors import NotFoundError
from app.domain.jobs import Job, JobState

JobWork = Callable[[str], Awaitable[None]]


class JobService:
  """Track asynchronous work and fan state changes out to subscribers."""

  def __init__(self, retention_seconds: int, initial_jobs: Iterable[Job] = ()) -> None:
    """Create a job registry and restore restart-spanning snapshots."""
    self.retention = timedelta(seconds=retention_seconds)
    self._jobs = {job.id: job for job in initial_jobs}
    self._tasks: dict[str, asyncio.Task[None]] = {}
    self._listeners: dict[str, set[asyncio.Queue[Job]]] = {}
    self._lock = asyncio.Lock()
    self._closing = False

  async def start(
    self,
    *,
    kind: str,
    resource_id: str,
    work: JobWork,
  ) -> Job:
    """Create a job and schedule its coroutine on the current event loop."""
    timestamp = datetime.now(UTC)
    job = Job(
      id=str(uuid.uuid4()),
      kind=kind,
      resource_id=resource_id,
      state=JobState.QUEUED,
      progress=0.0,
      stage="queued",
      detail="Waiting to begin",
      error=None,
      created_at=timestamp,
      updated_at=timestamp,
    )
    async with self._lock:
      if self._closing:
        raise RuntimeError("The job service is shutting down.")
      self._jobs[job.id] = job
      task = asyncio.create_task(self._run(job.id, work))
      self._tasks[job.id] = task
    return job

  async def _run(self, job_id: str, work: JobWork) -> None:
    """Run a scheduled job and preserve its terminal state."""
    await self.update(
      job_id,
      state=JobState.RUNNING,
      progress=0.01,
      stage="starting",
      detail="Starting background work",
    )
    try:
      await work(job_id)
    except asyncio.CancelledError:
      await self.update(
        job_id,
        state=JobState.CANCELLED,
        stage="cancelled",
        detail="Work was cancelled",
      )
    except Exception as error:
      await self.update(
        job_id,
        state=JobState.FAILED,
        stage="failed",
        detail="Work failed",
        error=str(error),
      )
    else:
      await self.update(
        job_id,
        state=JobState.COMPLETE,
        progress=1.0,
        stage="complete",
        detail="Finished successfully",
      )
    finally:
      async with self._lock:
        self._tasks.pop(job_id, None)

  async def update(
    self,
    job_id: str,
    *,
    state: JobState | None = None,
    progress: float | None = None,
    stage: str | None = None,
    detail: str | None = None,
    error: str | None = None,
  ) -> Job:
    """Update a job snapshot and notify subscribers without blocking work."""
    async with self._lock:
      current = self._jobs.get(job_id)
      if current is None:
        raise NotFoundError("Job")
      updated = replace(
        current,
        state=state or current.state,
        progress=max(0.0, min(progress, 1.0)) if progress is not None else current.progress,
        stage=stage or current.stage,
        detail=detail or current.detail,
        error=error,
        updated_at=datetime.now(UTC),
      )
      self._jobs[job_id] = updated
      listeners = tuple(self._listeners.get(job_id, ()))
    for queue in listeners:
      try:
        queue.put_nowait(updated)
      except asyncio.QueueFull:
        try:
          queue.get_nowait()
        except asyncio.QueueEmpty:
          pass
        queue.put_nowait(updated)
    return updated

  async def get(self, job_id: str) -> Job:
    """Return one retained job snapshot."""
    await self._prune()
    async with self._lock:
      job = self._jobs.get(job_id)
    if job is None:
      raise NotFoundError("Job")
    return job

  async def list(self) -> list[Job]:
    """List retained jobs newest first."""
    await self._prune()
    async with self._lock:
      return sorted(
        self._jobs.values(),
        key=lambda job: job.created_at,
        reverse=True,
      )

  async def cancel(self, job_id: str) -> Job:
    """Request cancellation for an active job."""
    async with self._lock:
      task = self._tasks.get(job_id)
    if task is None:
      return await self.get(job_id)
    task.cancel()
    try:
      await task
    except asyncio.CancelledError:
      pass
    return await self.get(job_id)

  async def shutdown(self) -> None:
    """Cancel every active job and wait for cancellation cleanup."""
    async with self._lock:
      self._closing = True
      tasks = tuple(self._tasks.values())
    for task in tasks:
      task.cancel()
    if tasks:
      await asyncio.gather(*tasks, return_exceptions=True)
    async with self._lock:
      self._listeners.clear()

  async def subscribe(self, job_id: str) -> AsyncGenerator[Job, None]:
    """Yield current and future snapshots until the job becomes terminal."""
    queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=8)
    initial = await self.get(job_id)
    async with self._lock:
      self._listeners.setdefault(job_id, set()).add(queue)
    try:
      yield initial
      if initial.terminal:
        return
      while True:
        snapshot = await queue.get()
        yield snapshot
        if snapshot.terminal:
          return
    finally:
      async with self._lock:
        listeners = self._listeners.get(job_id)
        if listeners is not None:
          listeners.discard(queue)
          if not listeners:
            self._listeners.pop(job_id, None)

  async def active_job_for(self, resource_id: str) -> Job | None:
    """Return the newest non-terminal job for a resource."""
    for job in await self.list():
      if job.resource_id == resource_id and not job.terminal:
        return job
    return None

  async def _prune(self) -> None:
    """Remove terminal snapshots after their configured retention window."""
    threshold = datetime.now(UTC) - self.retention
    async with self._lock:
      expired = [
        job_id for job_id, job in self._jobs.items() if job.terminal and job.updated_at < threshold
      ]
      for job_id in expired:
        self._jobs.pop(job_id, None)
        self._listeners.pop(job_id, None)
