"""Profile and sample workflows with staged, validated audio writes."""

from __future__ import annotations

import asyncio
import os
import tempfile
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import ValidationError
from app.domain.entities import Profile, StoredSample
from app.persistence.profiles import ProfileRepository
from app.services.style_controls import clean_style_controls
from app.utils.audio import (
  load_audio,
  load_audio_selection,
  prepare_reference_audio,
  write_wav,
)
from app.utils.files import promote_file, resolve_data_path, sha256_file, unlink_if_exists

CacheInvalidator = Callable[[str], Awaitable[int]]


def clean_profile_name(name: str) -> str:
  """Collapse whitespace and validate a human-readable profile name."""
  cleaned = " ".join(name.strip().split())
  if not cleaned:
    raise ValidationError("Profile name cannot be empty.")
  if len(cleaned) > 100:
    raise ValidationError("Profile name cannot exceed 100 characters.")
  return cleaned


def clean_transcript(transcript: str) -> str:
  """Trim and validate a reference transcript."""
  cleaned = transcript.strip()
  if not cleaned:
    raise ValidationError("Sample transcript cannot be empty.")
  if len(cleaned) > 5000:
    raise ValidationError("Sample transcript cannot exceed 5000 characters.")
  return cleaned


class ProfileService:
  """Coordinate profile persistence and sample file lifecycle."""

  def __init__(
    self,
    settings: Settings,
    repository: ProfileRepository,
    cache_invalidator: CacheInvalidator | None = None,
  ) -> None:
    """Store explicit configuration and persistence dependencies."""
    self.settings = settings
    self.repository = repository
    self.cache_invalidator = cache_invalidator

  async def _invalidate_cache(self, profile_id: str) -> None:
    """Invalidate derived voice prompts after sample-set mutations."""
    if self.cache_invalidator is not None:
      await self.cache_invalidator(profile_id)

  async def list_profiles(self, session: AsyncSession) -> list[Profile]:
    """List all profiles with aggregate sample counts."""
    return await self.repository.list_profiles(session)

  async def get_profile(self, session: AsyncSession, profile_id: str) -> Profile:
    """Return one profile."""
    return await self.repository.get_profile(session, profile_id)

  async def create_profile(self, session: AsyncSession, name: str) -> Profile:
    """Create a named profile with a generated stable ID."""
    return await self.repository.create_profile(
      session,
      profile_id=str(uuid.uuid4()),
      name=clean_profile_name(name),
    )

  async def rename_profile(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    name: str,
  ) -> Profile:
    """Rename a profile."""
    return await self.repository.rename_profile(
      session,
      profile_id=profile_id,
      name=clean_profile_name(name),
    )

  async def delete_profile(self, session: AsyncSession, profile_id: str) -> None:
    """Delete profile metadata and associated sample files."""
    samples = await self.repository.delete_profile(session, profile_id)
    await asyncio.gather(
      *(
        asyncio.to_thread(
          unlink_if_exists,
          resolve_data_path(self.settings.data_dir, sample.audio_file),
        )
        for sample in samples
      )
    )
    await self._invalidate_cache(profile_id)

  async def list_samples(
    self,
    session: AsyncSession,
    profile_id: str,
  ) -> list[StoredSample]:
    """List reference samples for a profile."""
    return await self.repository.list_samples(session, profile_id)

  async def get_sample(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    sample_id: str,
  ) -> StoredSample:
    """Return one reference sample."""
    return await self.repository.get_sample(
      session,
      profile_id=profile_id,
      sample_id=sample_id,
    )

  async def add_sample(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    source_path: Path,
    transcript: str,
    style_controls: tuple[str, ...] = (),
    sample_id: str | None = None,
    trim_start_seconds: float | None = None,
    trim_end_seconds: float | None = None,
  ) -> StoredSample:
    """Normalize a reference file, promote it, and persist its metadata."""
    await self.repository.get_profile(session, profile_id)
    resolved_id = sample_id or str(uuid.uuid4())
    clean_text = clean_transcript(transcript)
    clean_controls = clean_style_controls(style_controls)
    relative_path = (Path("audio") / "samples" / profile_id / f"{resolved_id}.wav").as_posix()
    destination = resolve_data_path(self.settings.data_dir, relative_path)

    def stage_audio() -> tuple[Path, int, int, str]:
      """Decode, validate, resample, and stage one WAV file."""
      if trim_start_seconds is None and trim_end_seconds is None:
        audio = load_audio(source_path, sample_rate=None)
      else:
        audio, _ = load_audio_selection(
          source_path,
          start_seconds=trim_start_seconds,
          end_seconds=trim_end_seconds,
          sample_rate=None,
        )
      audio = prepare_reference_audio(audio)
      destination.parent.mkdir(parents=True, exist_ok=True)
      descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{resolved_id}-",
        suffix=".wav",
        dir=destination.parent,
      )
      os.close(descriptor)
      staged = Path(staged_name)
      try:
        write_wav(staged, audio)
        promote_file(staged, destination)
      finally:
        unlink_if_exists(staged)
      return (
        destination,
        audio.duration_ms,
        audio.sample_rate,
        sha256_file(destination),
      )

    destination_path, duration_ms, sample_rate, digest = await asyncio.to_thread(stage_audio)
    sample = StoredSample(
      id=resolved_id,
      profile_id=profile_id,
      transcript=clean_text,
      style_controls=clean_controls,
      duration_ms=duration_ms,
      sample_rate=sample_rate,
      content_sha256=digest,
      created_at=datetime.now(UTC),
      audio_file=relative_path,
    )
    try:
      created = await self.repository.create_sample(session, sample=sample)
    except Exception:
      await asyncio.to_thread(unlink_if_exists, destination_path)
      raise
    await self._invalidate_cache(profile_id)
    return created

  async def update_sample_metadata(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    sample_id: str,
    transcript: str,
    style_controls: tuple[str, ...] | None = None,
  ) -> StoredSample:
    """Update a sample transcript and optional conditioning metadata."""
    clean_controls = None
    if style_controls is not None:
      clean_controls = clean_style_controls(style_controls)
    updated = await self.repository.update_sample_metadata(
      session,
      profile_id=profile_id,
      sample_id=sample_id,
      transcript=clean_transcript(transcript),
      style_controls=clean_controls,
    )
    await self._invalidate_cache(profile_id)
    return updated

  async def delete_sample(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    sample_id: str,
  ) -> None:
    """Delete sample metadata and its normalized audio."""
    sample = await self.repository.delete_sample(
      session,
      profile_id=profile_id,
      sample_id=sample_id,
    )
    path = resolve_data_path(self.settings.data_dir, sample.audio_file)
    await asyncio.to_thread(unlink_if_exists, path)
    await self._invalidate_cache(profile_id)

  def sample_path(self, sample: StoredSample) -> Path:
    """Resolve a sample's private relative audio path."""
    return resolve_data_path(self.settings.data_dir, sample.audio_file)
