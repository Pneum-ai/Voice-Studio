"""Profile and sample repository with aggregate list queries."""

from __future__ import annotations

import json
from collections.abc import Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import ConflictError, NotFoundError
from app.domain.entities import Profile, StoredSample
from app.persistence.orm import ProfileRecord, ProfileSampleRecord
from app.persistence.time import datetime_from_ms, now_ms


def normalize_profile_name(name: str) -> str:
  """Normalize a profile name for case-insensitive uniqueness checks."""
  return " ".join(name.strip().split()).casefold()


def _profile_entity(record: ProfileRecord, sample_count: int) -> Profile:
  """Convert a profile mapping and aggregate count to a domain entity."""
  return Profile(
    id=record.id,
    name=record.name,
    sample_count=sample_count,
    created_at=datetime_from_ms(record.created_at_ms),
    updated_at=datetime_from_ms(record.updated_at_ms),
  )


def _sample_entity(record: ProfileSampleRecord) -> StoredSample:
  """Convert a sample mapping to a private storage entity."""
  return StoredSample(
    id=record.id,
    profile_id=record.profile_id,
    transcript=record.transcript,
    style_controls=tuple(json.loads(record.style_controls_json)),
    duration_ms=record.duration_ms,
    sample_rate=record.sample_rate,
    content_sha256=record.content_sha256,
    created_at=datetime_from_ms(record.created_at_ms),
    audio_file=record.audio_file,
  )


class ProfileRepository:
  """Read and mutate profiles and samples in short transactions."""

  async def list_profiles(self, session: AsyncSession) -> list[Profile]:
    """List profiles newest first with sample counts in one query."""
    count = func.count(ProfileSampleRecord.id)
    statement = (
      select(ProfileRecord, count)
      .outerjoin(ProfileSampleRecord, ProfileSampleRecord.profile_id == ProfileRecord.id)
      .group_by(ProfileRecord.id)
      .order_by(ProfileRecord.created_at_ms.desc())
    )
    rows = (await session.execute(statement)).all()
    return [_profile_entity(record, sample_count) for record, sample_count in rows]

  async def get_profile(self, session: AsyncSession, profile_id: str) -> Profile:
    """Return one profile with its aggregate sample count."""
    count = func.count(ProfileSampleRecord.id)
    statement = (
      select(ProfileRecord, count)
      .outerjoin(ProfileSampleRecord, ProfileSampleRecord.profile_id == ProfileRecord.id)
      .where(ProfileRecord.id == profile_id)
      .group_by(ProfileRecord.id)
    )
    row = (await session.execute(statement)).one_or_none()
    if row is None:
      raise NotFoundError("Profile")
    return _profile_entity(row[0], row[1])

  async def create_profile(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    name: str,
  ) -> Profile:
    """Create a uniquely named profile and commit it."""
    timestamp = now_ms()
    record = ProfileRecord(
      id=profile_id,
      name=name,
      name_key=normalize_profile_name(name),
      created_at_ms=timestamp,
      updated_at_ms=timestamp,
    )
    session.add(record)
    try:
      await session.commit()
    except IntegrityError as error:
      await session.rollback()
      raise ConflictError(f"A profile named {name!r} already exists.") from error
    return _profile_entity(record, 0)

  async def rename_profile(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    name: str,
  ) -> Profile:
    """Rename a profile while preserving normalized uniqueness."""
    record = await session.get(ProfileRecord, profile_id)
    if record is None:
      raise NotFoundError("Profile")
    record.name = name
    record.name_key = normalize_profile_name(name)
    record.updated_at_ms = now_ms()
    try:
      await session.commit()
    except IntegrityError as error:
      await session.rollback()
      raise ConflictError(f"A profile named {name!r} already exists.") from error
    return await self.get_profile(session, profile_id)

  async def delete_profile(
    self,
    session: AsyncSession,
    profile_id: str,
  ) -> list[StoredSample]:
    """Delete a profile and return sample paths for post-commit cleanup."""
    await self.get_profile(session, profile_id)
    samples = await self.list_samples(session, profile_id)
    await session.execute(delete(ProfileRecord).where(ProfileRecord.id == profile_id))
    await session.commit()
    return samples

  async def list_samples(
    self,
    session: AsyncSession,
    profile_id: str,
  ) -> list[StoredSample]:
    """List reference samples in creation order."""
    await self.get_profile(session, profile_id)
    statement = (
      select(ProfileSampleRecord)
      .where(ProfileSampleRecord.profile_id == profile_id)
      .order_by(ProfileSampleRecord.created_at_ms.asc())
    )
    records: Sequence[ProfileSampleRecord] = (await session.scalars(statement)).all()
    return [_sample_entity(record) for record in records]

  async def get_sample(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    sample_id: str,
  ) -> StoredSample:
    """Return one sample scoped to its parent profile."""
    statement = select(ProfileSampleRecord).where(
      ProfileSampleRecord.id == sample_id,
      ProfileSampleRecord.profile_id == profile_id,
    )
    record = (await session.scalars(statement)).one_or_none()
    if record is None:
      raise NotFoundError("Profile sample")
    return _sample_entity(record)

  async def create_sample(
    self,
    session: AsyncSession,
    *,
    sample: StoredSample,
  ) -> StoredSample:
    """Create sample metadata after its staged audio has been promoted."""
    await self.get_profile(session, sample.profile_id)
    record = ProfileSampleRecord(
      id=sample.id,
      profile_id=sample.profile_id,
      audio_file=sample.audio_file,
      transcript=sample.transcript,
      style_controls_json=json.dumps(sample.style_controls),
      duration_ms=sample.duration_ms,
      sample_rate=sample.sample_rate,
      content_sha256=sample.content_sha256,
      created_at_ms=int(sample.created_at.timestamp() * 1000),
    )
    session.add(record)
    profile = await session.get(ProfileRecord, sample.profile_id)
    if profile is not None:
      profile.updated_at_ms = now_ms()
    try:
      await session.commit()
    except IntegrityError as error:
      await session.rollback()
      raise ConflictError("A sample with this ID already exists.") from error
    return _sample_entity(record)

  async def update_sample_metadata(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    sample_id: str,
    transcript: str,
    style_controls: tuple[str, ...] | None,
  ) -> StoredSample:
    """Update a sample transcript, optional style controls, and modification time."""
    await self.get_sample(session, profile_id=profile_id, sample_id=sample_id)
    record = await session.get(ProfileSampleRecord, sample_id)
    assert record is not None
    record.transcript = transcript
    if style_controls is not None:
      record.style_controls_json = json.dumps(style_controls)
    profile = await session.get(ProfileRecord, profile_id)
    if profile is not None:
      profile.updated_at_ms = now_ms()
    await session.commit()
    return _sample_entity(record)

  async def delete_sample(
    self,
    session: AsyncSession,
    *,
    profile_id: str,
    sample_id: str,
  ) -> StoredSample:
    """Delete sample metadata and return its path for file cleanup."""
    sample = await self.get_sample(
      session,
      profile_id=profile_id,
      sample_id=sample_id,
    )
    await session.execute(delete(ProfileSampleRecord).where(ProfileSampleRecord.id == sample_id))
    profile = await session.get(ProfileRecord, profile_id)
    if profile is not None:
      profile.updated_at_ms = now_ms()
    await session.commit()
    return sample
