"""Saved generation metadata and audio file lifecycle."""

from __future__ import annotations

import asyncio
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.domain.entities import StoredGeneration
from app.persistence.generations import GenerationRepository
from app.utils.audio import AudioData, write_wav
from app.utils.files import promote_file, resolve_data_path, sha256_file, unlink_if_exists


class GenerationService:
  """Persist, list, resolve, and delete generated speech."""

  def __init__(self, settings: Settings, repository: GenerationRepository) -> None:
    """Store explicit settings and repository dependencies."""
    self.settings = settings
    self.repository = repository

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
    """List saved generations using cursor pagination."""
    return await self.repository.list_generations(
      session,
      limit=limit,
      cursor=cursor,
      profile_id=profile_id,
      model=model,
      query=query,
    )

  async def get_generation(
    self,
    session: AsyncSession,
    generation_id: str,
  ) -> StoredGeneration:
    """Return one saved generation."""
    return await self.repository.get_generation(session, generation_id)

  async def save_generation(
    self,
    session: AsyncSession,
    *,
    profile_id: str | None,
    profile_name: str,
    input_text: str,
    language: str,
    model: str,
    seed: int | None,
    instructions: str | None,
    audio: AudioData,
  ) -> StoredGeneration:
    """Stage a WAV, promote it, and persist complete generation metadata."""
    generation_id = str(uuid.uuid4())
    relative_path = (Path("audio") / "generations" / f"{generation_id}.wav").as_posix()
    destination = resolve_data_path(self.settings.data_dir, relative_path)

    def stage_audio() -> str:
      """Write and atomically promote one generated WAV."""
      destination.parent.mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(
        prefix=f".{generation_id}-",
        suffix=".wav",
        dir=destination.parent,
        delete=False,
      ) as temporary:
        staged = Path(temporary.name)
      try:
        write_wav(staged, audio)
        promote_file(staged, destination)
      finally:
        unlink_if_exists(staged)
      return sha256_file(destination)

    digest = await asyncio.to_thread(stage_audio)
    generation = StoredGeneration(
      id=generation_id,
      profile_id=profile_id,
      profile_name=profile_name,
      input_text=input_text,
      language=language,
      model=model,
      seed=seed,
      instructions=instructions,
      duration_ms=audio.duration_ms,
      sample_rate=audio.sample_rate,
      created_at=datetime.now(UTC),
      audio_file=relative_path,
      audio_sha256=digest,
    )
    try:
      return await self.repository.create_generation(session, generation)
    except Exception:
      await asyncio.to_thread(unlink_if_exists, destination)
      raise

  async def delete_generation(
    self,
    session: AsyncSession,
    generation_id: str,
  ) -> None:
    """Delete generation metadata and its WAV."""
    generation = await self.repository.delete_generation(session, generation_id)
    path = resolve_data_path(self.settings.data_dir, generation.audio_file)
    await asyncio.to_thread(unlink_if_exists, path)

  def audio_path(self, generation: StoredGeneration) -> Path:
    """Resolve a saved generation's private relative audio path."""
    return resolve_data_path(self.settings.data_dir, generation.audio_file)
