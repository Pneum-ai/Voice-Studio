"""Saved generation repository with cursor-based list queries."""

from __future__ import annotations

import base64
from collections.abc import Sequence

from sqlalchemy import column, delete, or_, select, table
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.errors import NotFoundError, ValidationError
from app.domain.entities import StoredGeneration
from app.persistence.orm import GenerationRecord
from app.persistence.time import datetime_from_ms

GENERATIONS_FTS = table(
  "generations_fts",
  column("generation_id"),
  column("input_text"),
)


def encode_cursor(timestamp_ms: int, generation_id: str) -> str:
  """Encode a stable timestamp and ID pagination cursor."""
  value = f"{timestamp_ms}:{generation_id}".encode()
  return base64.urlsafe_b64encode(value).decode().rstrip("=")


def decode_cursor(cursor: str) -> tuple[int, str]:
  """Decode a pagination cursor or raise a public validation error."""
  try:
    padded = cursor + "=" * (-len(cursor) % 4)
    timestamp, generation_id = base64.urlsafe_b64decode(padded).decode().split(":", 1)
    return int(timestamp), generation_id
  except (ValueError, UnicodeDecodeError) as error:
    raise ValidationError("The generation cursor is invalid.") from error


def generation_entity(record: GenerationRecord) -> StoredGeneration:
  """Convert a generation mapping to a private storage entity."""
  return StoredGeneration(
    id=record.id,
    profile_id=record.profile_id,
    profile_name=record.profile_name,
    input_text=record.input_text,
    language=record.language,
    model=record.model,
    seed=record.seed,
    instructions=record.instructions,
    duration_ms=record.duration_ms,
    sample_rate=record.sample_rate,
    created_at=datetime_from_ms(record.created_at_ms),
    audio_file=record.audio_file,
    audio_sha256=record.audio_sha256,
  )


class GenerationRepository:
  """Read and mutate persisted generations."""

  async def list_generations(
    self,
    session: AsyncSession,
    *,
    limit: int,
    cursor: str | None,
    profile_id: str | None,
    model: str | None,
    query: str | None,
  ) -> tuple[list[StoredGeneration], str | None]:
    """List saved generations with stable keyset pagination."""
    statement = select(GenerationRecord)
    if profile_id:
      statement = statement.where(GenerationRecord.profile_id == profile_id)
    if model:
      statement = statement.where(GenerationRecord.model == model)
    if query:
      escaped_query = query.replace('"', '""')
      matches = select(GENERATIONS_FTS.c.generation_id).where(
        GENERATIONS_FTS.c.input_text.match(f'"{escaped_query}"')
      )
      statement = statement.where(GenerationRecord.id.in_(matches))
    if cursor:
      timestamp, generation_id = decode_cursor(cursor)
      statement = statement.where(
        or_(
          GenerationRecord.created_at_ms < timestamp,
          ((GenerationRecord.created_at_ms == timestamp) & (GenerationRecord.id < generation_id)),
        )
      )
    statement = statement.order_by(
      GenerationRecord.created_at_ms.desc(),
      GenerationRecord.id.desc(),
    ).limit(limit + 1)
    records: Sequence[GenerationRecord] = (await session.scalars(statement)).all()
    page = list(records[:limit])
    next_cursor = None
    if len(records) > limit and page:
      last = page[-1]
      next_cursor = encode_cursor(last.created_at_ms, last.id)
    return [generation_entity(record) for record in page], next_cursor

  async def get_generation(
    self,
    session: AsyncSession,
    generation_id: str,
  ) -> StoredGeneration:
    """Return one saved generation."""
    record = await session.get(GenerationRecord, generation_id)
    if record is None:
      raise NotFoundError("Generation")
    return generation_entity(record)

  async def create_generation(
    self,
    session: AsyncSession,
    generation: StoredGeneration,
  ) -> StoredGeneration:
    """Persist generation metadata after its WAV has been promoted."""
    record = GenerationRecord(
      id=generation.id,
      profile_id=generation.profile_id,
      profile_name=generation.profile_name,
      input_text=generation.input_text,
      language=generation.language,
      model=generation.model,
      seed=generation.seed,
      instructions=generation.instructions,
      audio_file=generation.audio_file,
      audio_sha256=generation.audio_sha256,
      duration_ms=generation.duration_ms,
      sample_rate=generation.sample_rate,
      created_at_ms=int(generation.created_at.timestamp() * 1000),
    )
    session.add(record)
    await session.commit()
    return generation_entity(record)

  async def delete_generation(
    self,
    session: AsyncSession,
    generation_id: str,
  ) -> StoredGeneration:
    """Delete generation metadata and return its audio path for cleanup."""
    generation = await self.get_generation(session, generation_id)
    await session.execute(delete(GenerationRecord).where(GenerationRecord.id == generation_id))
    await session.commit()
    return generation
