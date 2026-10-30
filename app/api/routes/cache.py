"""Derived voice-prompt cache management route."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.dependencies import ServiceDependency
from app.api.schemas.common import CacheClearResponse

router = APIRouter(prefix="/v1/cache", tags=["cache"])


@router.delete(
  "/profile-prompts",
  response_model=CacheClearResponse,
  summary="Clear derived profile prompts",
)
async def clear_profile_prompts(services: ServiceDependency) -> CacheClearResponse:
  """Clear safe prompt tensors and concatenated reference WAV files."""
  deleted = await services.prompts.clear_all()
  return CacheClearResponse(deleted_files=deleted)
