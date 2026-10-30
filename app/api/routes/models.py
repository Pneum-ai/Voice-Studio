"""Model catalog and desired-state routes."""

from __future__ import annotations

from fastapi import APIRouter, status

from app.api.dependencies import ServiceDependency
from app.api.schemas.common import ProblemResponse
from app.api.schemas.models import (
  JobResponse,
  ModelListResponse,
  ModelResponse,
  ModelStateRequest,
)

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get(
  "",
  response_model=ModelListResponse,
  summary="List selectable models",
)
async def list_models(services: ServiceDependency) -> ModelListResponse:
  """Return speech and transcription model states plus capabilities."""
  models = await services.models.list_models()
  return ModelListResponse(items=[ModelResponse.from_entity(model) for model in models])


@router.get(
  "/{model_id}",
  response_model=ModelResponse,
  responses={400: {"model": ProblemResponse}},
  summary="Get a model",
)
async def get_model(
  model_id: str,
  services: ServiceDependency,
) -> ModelResponse:
  """Return one catalog model and its current local state."""
  model = await services.models.get_model(model_id)
  return ModelResponse.from_entity(model)


@router.patch(
  "/{model_id}",
  response_model=JobResponse,
  status_code=status.HTTP_202_ACCEPTED,
  responses={
    400: {"model": ProblemResponse},
    409: {"model": ProblemResponse},
  },
  summary="Set a desired model state",
)
async def update_model(
  model_id: str,
  payload: ModelStateRequest,
  services: ServiceDependency,
) -> JobResponse:
  """Start an asynchronous download, load, unload, or removal transition."""
  job = await services.models.request_state(model_id, payload.state)
  return JobResponse.from_entity(job)
