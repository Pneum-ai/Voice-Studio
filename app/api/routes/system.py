"""Unversioned health and API-only root routes."""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from fastapi.responses import FileResponse

from app import __version__
from app.api.dependencies import ServiceDependency
from app.api.schemas.common import (
  HealthResponse,
  InferenceReadinessResponse,
  ServiceMetadataResponse,
)

router = APIRouter(tags=["system"])


@router.get(
  "/health",
  response_model=HealthResponse,
  responses={
    503: {
      "model": HealthResponse,
      "description": "The API is live but accelerated inference is unavailable.",
    }
  },
  summary="Check process readiness",
)
async def health(response: Response, services: ServiceDependency) -> HealthResponse:
  """Return dependency readiness without loading or downloading models."""
  runtime_status = getattr(services.backends.tts, "runtime_status", None)
  inference = (
    InferenceReadinessResponse(
      ready=runtime_status.ready,
      reason=runtime_status.reason,
      active=runtime_status.active,
      queued=runtime_status.queued,
      active_limit=runtime_status.active_limit,
      active_ceiling=runtime_status.active_ceiling,
      failure_id=runtime_status.failure_id,
      failure_detail=runtime_status.failure_detail,
      policy_name=getattr(runtime_status, "policy_name", None),
      capture_profile=getattr(runtime_status, "capture_profile", None),
      capture_profile_sha256=getattr(runtime_status, "capture_profile_sha256", None),
      capture_fallbacks=getattr(runtime_status, "capture_fallbacks", 0),
    )
    if runtime_status is not None
    else None
  )
  if inference is not None and not inference.ready:
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
  return HealthResponse(
    status="ready" if inference is None or inference.ready else "unavailable",
    version=__version__,
    backend=services.backends.name,
    device=services.backends.device,
    device_name=services.backends.device_info.device_name,
    device_advisory=services.backends.device_info.accelerator_advisory,
    runtime_version=services.backends.device_info.runtime_version,
    ui_enabled=services.settings.ui_enabled,
    inference=inference,
  )


@router.get(
  "/",
  response_model=None,
  summary="Describe the local API",
)
async def root(
  services: ServiceDependency,
) -> ServiceMetadataResponse | FileResponse:
  """Return service links, or the bundled application shell in UI mode."""
  if services.settings.ui_enabled:
    return FileResponse(
      services.settings.ui_dist_dir / "index.html",
      media_type="text/html",
      headers={"Cache-Control": "no-cache"},
    )
  return ServiceMetadataResponse(version=__version__)
