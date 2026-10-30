"""Typed domain entities independent of HTTP and database implementations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Profile:
  """Represent a named voice profile."""

  id: str
  name: str
  sample_count: int
  created_at: datetime
  updated_at: datetime


@dataclass(frozen=True, slots=True)
class ProfileSample:
  """Represent one normalized voice reference sample."""

  id: str
  profile_id: str
  transcript: str
  style_controls: tuple[str, ...]
  duration_ms: int
  sample_rate: int
  content_sha256: str
  created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredSample(ProfileSample):
  """Represent a sample together with its private relative audio path."""

  audio_file: str


@dataclass(frozen=True, slots=True)
class Generation:
  """Represent a persisted speech generation and its complete input metadata."""

  id: str
  profile_id: str | None
  profile_name: str
  input_text: str
  language: str
  model: str
  seed: int | None
  instructions: str | None
  duration_ms: int
  sample_rate: int
  created_at: datetime


@dataclass(frozen=True, slots=True)
class StoredGeneration(Generation):
  """Represent a generation together with private storage metadata."""

  audio_file: str
  audio_sha256: str
