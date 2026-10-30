"""Voice profile and reference sample API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.domain.entities import Profile, StoredSample


class ProfileCreateRequest(BaseModel):
  """Request creation of an empty named voice profile."""

  name: str = Field(min_length=1, max_length=100)


class ProfileUpdateRequest(BaseModel):
  """Request a profile rename."""

  name: str = Field(min_length=1, max_length=100)


class ProfileResponse(BaseModel):
  """Expose public profile metadata."""

  model_config = ConfigDict(from_attributes=True)

  id: str
  name: str
  sample_count: int
  created_at: datetime
  updated_at: datetime

  @classmethod
  def from_entity(cls, profile: Profile) -> ProfileResponse:
    """Convert a profile domain entity into a public response."""
    return cls.model_validate(profile)


class ProfileListResponse(BaseModel):
  """Wrap a deterministic profile collection."""

  items: list[ProfileResponse]


class SampleUpdateRequest(BaseModel):
  """Request reference transcript and optional style metadata replacement."""

  transcript: str = Field(min_length=1, max_length=5000)
  style_controls: tuple[str, ...] | None = Field(default=None, max_length=5)


class SampleResponse(BaseModel):
  """Expose reference metadata without private filesystem paths."""

  id: str
  profile_id: str
  transcript: str
  style_controls: tuple[str, ...]
  duration_seconds: float
  sample_rate: int
  sha256: str
  audio_url: str
  created_at: datetime

  @classmethod
  def from_entity(cls, sample: StoredSample) -> SampleResponse:
    """Convert a stored sample into its public URL-based representation."""
    return cls(
      id=sample.id,
      profile_id=sample.profile_id,
      transcript=sample.transcript,
      style_controls=sample.style_controls,
      duration_seconds=sample.duration_ms / 1000,
      sample_rate=sample.sample_rate,
      sha256=sample.content_sha256,
      audio_url=f"/v1/profiles/{sample.profile_id}/samples/{sample.id}/audio",
      created_at=sample.created_at,
    )


class SampleListResponse(BaseModel):
  """Wrap a profile's ordered sample collection."""

  items: list[SampleResponse]
