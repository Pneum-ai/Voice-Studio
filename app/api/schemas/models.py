"""Model lifecycle and asynchronous job API schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.jobs import Job
from app.domain.models import (
  DesiredModelState,
  ModelStatus,
  TranscriptionCapabilities,
)


class TranscriptionCapabilitiesResponse(BaseModel):
  """Expose model-specific recognition inputs and optional features."""

  languages: list[str]
  language_mode: Literal["automatic", "optional", "required"]
  context: bool
  timestamps: list[str]
  streaming: bool
  translation: bool
  max_audio_seconds: int | None

  @classmethod
  def from_entity(
    cls,
    capabilities: TranscriptionCapabilities,
  ) -> TranscriptionCapabilitiesResponse:
    """Convert immutable domain capabilities into an API response."""
    return cls(
      languages=list(capabilities.languages),
      language_mode=capabilities.language_mode.value,
      context=capabilities.context,
      timestamps=list(capabilities.timestamps),
      streaming=capabilities.streaming,
      translation=capabilities.translation,
      max_audio_seconds=capabilities.max_audio_seconds,
    )


class SpeechCapabilitiesResponse(BaseModel):
  """Expose valid voice sources and instruction support for a speech model."""

  voice_mode: Literal["clone", "builtin"]
  speakers: list[str]
  instructions: bool


class ModelResponse(BaseModel):
  """Expose model availability, memory state, and backend capabilities."""

  id: str
  display_name: str
  kind: str
  family: str
  provider: str
  transcription: TranscriptionCapabilitiesResponse | None
  speech: SpeechCapabilitiesResponse | None
  download_state: Literal["absent", "downloaded"]
  load_state: Literal["loaded", "unloaded"]
  size_bytes: int | None
  backend_supported: bool
  native_streaming: bool
  active_job_id: str | None

  @classmethod
  def from_entity(cls, model: ModelStatus) -> ModelResponse:
    """Convert an internal model snapshot into a public response."""
    return cls(
      id=model.id,
      display_name=model.display_name,
      kind=model.kind.value,
      family=model.family,
      provider=model.provider,
      transcription=(
        TranscriptionCapabilitiesResponse.from_entity(model.transcription)
        if model.transcription
        else None
      ),
      speech=(
        SpeechCapabilitiesResponse(
          voice_mode=model.speech.voice_mode,
          speakers=list(model.speech.speakers),
          instructions=model.speech.instructions,
        )
        if model.speech
        else None
      ),
      download_state="downloaded" if model.downloaded else "absent",
      load_state="loaded" if model.loaded else "unloaded",
      size_bytes=model.size_bytes,
      backend_supported=model.backend_supported,
      native_streaming=model.native_streaming,
      active_job_id=model.active_job_id,
    )


class ModelListResponse(BaseModel):
  """Wrap the complete selectable model catalog."""

  items: list[ModelResponse]


class ModelStateRequest(BaseModel):
  """Request a desired local state for one model."""

  state: DesiredModelState


class JobResponse(BaseModel):
  """Expose one asynchronous lifecycle job snapshot."""

  id: str
  kind: str
  resource_id: str
  state: str
  progress: float = Field(ge=0, le=1)
  stage: str
  detail: str
  error: str | None
  created_at: datetime
  updated_at: datetime

  @classmethod
  def from_entity(cls, job: Job) -> JobResponse:
    """Convert an internal job snapshot into a public response."""
    return cls(
      id=job.id,
      kind=job.kind,
      resource_id=job.resource_id,
      state=job.state.value,
      progress=job.progress,
      stage=job.stage,
      detail=job.detail,
      error=job.error,
      created_at=job.created_at,
      updated_at=job.updated_at,
    )


class JobListResponse(BaseModel):
  """Wrap retained jobs newest first."""

  items: list[JobResponse]
