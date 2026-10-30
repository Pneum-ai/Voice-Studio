"""Shared system and RFC-style problem response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProblemResponse(BaseModel):
  """Describe one stable API failure envelope."""

  type: str = "about:blank"
  title: str
  status: int
  code: str
  detail: str
  request_id: str
  errors: list[dict[str, Any]] | None = None


class HealthResponse(BaseModel):
  """Describe process readiness without mutating model state."""

  status: str = "ready"
  version: str
  backend: str
  device: str
  device_name: str | None = None
  device_advisory: str | None = None
  runtime_version: str | None = None
  ui_enabled: bool
  inference: InferenceReadinessResponse | None = None


class InferenceReadinessResponse(BaseModel):
  """Expose accelerated inference readiness and admission pressure."""

  ready: bool
  reason: str
  active: int = Field(ge=0)
  queued: int = Field(ge=0)
  active_limit: int = Field(ge=1)
  active_ceiling: int = Field(ge=1)
  failure_id: str | None = None
  failure_detail: str | None = None
  policy_name: str | None = None
  capture_profile: str | None = None
  capture_profile_sha256: str | None = None
  capture_fallbacks: int = Field(default=0, ge=0)


class ServiceMetadataResponse(BaseModel):
  """Describe the API-only root resource."""

  name: str = "Pneum.ai Voice Studio"
  version: str
  api: str = "/v1"
  docs: str = "/docs"


class DeleteResponse(BaseModel):
  """Confirm deletion of one resource."""

  deleted: bool = True


class CacheClearResponse(BaseModel):
  """Report the number of derived cache files removed."""

  deleted_files: int = Field(ge=0)
