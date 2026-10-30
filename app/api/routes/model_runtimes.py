"""Optional model-runtime installation routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.dependencies import ServiceDependency
from app.api.schemas.common import ProblemResponse
from app.api.schemas.models import JobResponse

router = APIRouter(prefix="/v1/model-runtimes", tags=["models"])


@router.post(
  "/install-all",
  response_model=JobResponse,
  status_code=status.HTTP_202_ACCEPTED,
  responses={409: {"model": ProblemResponse}},
  summary="Install compatible transcription runtimes",
)
async def install_all_runtimes(services: ServiceDependency) -> JobResponse:
  """Install optional Qwen3-ASR and NVIDIA ASR dependencies as a retained job."""
  job = await services.model_runtimes.install_all()
  return JobResponse.from_entity(job)
