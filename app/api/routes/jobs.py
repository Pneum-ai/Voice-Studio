"""Retained model-job snapshots and server-sent event progress."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.dependencies import ServiceDependency
from app.api.schemas.common import ProblemResponse
from app.api.schemas.models import JobListResponse, JobResponse

router = APIRouter(prefix="/v1/jobs", tags=["jobs"])


@router.get(
  "",
  response_model=JobListResponse,
  summary="List retained jobs",
)
async def list_jobs(services: ServiceDependency) -> JobListResponse:
  """List active and recently completed jobs newest first."""
  jobs = await services.jobs.list()
  return JobListResponse(items=[JobResponse.from_entity(job) for job in jobs])


@router.get(
  "/{job_id}",
  response_model=JobResponse,
  responses={404: {"model": ProblemResponse}},
  summary="Get a job",
)
async def get_job(job_id: str, services: ServiceDependency) -> JobResponse:
  """Return one retained job snapshot."""
  return JobResponse.from_entity(await services.jobs.get(job_id))


@router.get(
  "/{job_id}/events",
  response_class=StreamingResponse,
  responses={404: {"model": ProblemResponse}},
  summary="Follow job progress",
)
async def stream_job_events(
  job_id: str,
  services: ServiceDependency,
) -> StreamingResponse:
  """Stream job snapshots over SSE with idle heartbeat comments."""
  await services.jobs.get(job_id)

  async def events() -> AsyncIterator[str]:
    """Yield JSON job events until a terminal snapshot is observed."""
    iterator = services.jobs.subscribe(job_id)
    pending = asyncio.ensure_future(anext(iterator))
    try:
      while True:
        done, _ = await asyncio.wait({pending}, timeout=15)
        if not done:
          yield ": heartbeat\n\n"
          continue
        try:
          snapshot = pending.result()
        except StopAsyncIteration:
          return
        response = JobResponse.from_entity(snapshot)
        yield f"event: {snapshot.state.value}\ndata: {response.model_dump_json()}\n\n"
        if snapshot.terminal:
          return
        pending = asyncio.ensure_future(anext(iterator))
    finally:
      if not pending.done():
        pending.cancel()
      await iterator.aclose()

  return StreamingResponse(
    events(),
    media_type="text/event-stream",
    headers={
      "Cache-Control": "no-cache",
      "X-Accel-Buffering": "no",
    },
  )
