"""Saved speech generation API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from app.domain.entities import StoredGeneration


class GenerationResponse(BaseModel):
  """Expose persisted speech metadata and a stable audio URL."""

  id: str
  profile_id: str | None
  profile_name: str
  input_text: str
  language: str
  model: str
  seed: int | None
  instructions: str | None
  duration_seconds: float
  sample_rate: int
  sha256: str
  audio_url: str
  created_at: datetime

  @classmethod
  def from_entity(cls, generation: StoredGeneration) -> GenerationResponse:
    """Convert a private stored generation into a public response."""
    return cls(
      id=generation.id,
      profile_id=generation.profile_id,
      profile_name=generation.profile_name,
      input_text=generation.input_text,
      language=generation.language,
      model=generation.model,
      seed=generation.seed,
      instructions=generation.instructions,
      duration_seconds=generation.duration_ms / 1000,
      sample_rate=generation.sample_rate,
      sha256=generation.audio_sha256,
      audio_url=f"/v1/generations/{generation.id}/audio",
      created_at=generation.created_at,
    )


class GenerationListResponse(BaseModel):
  """Wrap one cursor-paginated generation page."""

  items: list[GenerationResponse]
  next_cursor: str | None
