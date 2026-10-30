"""Ordered migrations for the independent fresh SQLite schema."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncConnection

from app.persistence.orm import Base

MigrationWork = Callable[[AsyncConnection], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Migration:
  """Describe one ordered schema transition."""

  version: int
  name: str
  apply: MigrationWork


async def create_initial_schema(connection: AsyncConnection) -> None:
  """Create the initial optimized relational tables."""
  await connection.run_sync(Base.metadata.create_all)


async def add_generation_search(connection: AsyncConnection) -> None:
  """Add model filtering and synchronized generated-text search indexes."""
  for statement in (
    """
    CREATE INDEX IF NOT EXISTS ix_generations_model_created
    ON generations (model, created_at_ms)
    """,
    """
    CREATE VIRTUAL TABLE IF NOT EXISTS generations_fts
    USING fts5(generation_id UNINDEXED, input_text, tokenize='trigram')
    """,
    """
    CREATE TRIGGER IF NOT EXISTS generations_fts_insert
    AFTER INSERT ON generations BEGIN
      INSERT INTO generations_fts(generation_id, input_text)
      VALUES (new.id, new.input_text);
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS generations_fts_delete
    AFTER DELETE ON generations BEGIN
      DELETE FROM generations_fts WHERE generation_id = old.id;
    END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS generations_fts_update
    AFTER UPDATE OF input_text ON generations BEGIN
      DELETE FROM generations_fts WHERE generation_id = old.id;
      INSERT INTO generations_fts(generation_id, input_text)
      VALUES (new.id, new.input_text);
    END
    """,
  ):
    await connection.execute(text(statement))


async def preserve_schema_version_three(_: AsyncConnection) -> None:
  """Reserve a removed feature's historical schema version for compatibility."""


async def preserve_schema_version_four(_: AsyncConnection) -> None:
  """Reserve a removed feature's historical schema version for compatibility."""


async def add_sample_style_controls(connection: AsyncConnection) -> None:
  """Add optional conditioning metadata to reference samples."""
  table_info = await connection.execute(text("PRAGMA table_info(profile_samples)"))
  columns = {str(row[1]) for row in table_info}
  if "style_controls_json" not in columns:
    await connection.execute(
      text("ALTER TABLE profile_samples ADD COLUMN style_controls_json TEXT NOT NULL DEFAULT '[]'")
    )


MIGRATIONS = (
  Migration(1, "create_initial_schema", create_initial_schema),
  Migration(2, "add_generation_search", add_generation_search),
  Migration(3, "preserve_schema_version_three", preserve_schema_version_three),
  Migration(4, "preserve_schema_version_four", preserve_schema_version_four),
  Migration(5, "add_sample_style_controls", add_sample_style_controls),
)
LATEST_SCHEMA_VERSION = MIGRATIONS[-1].version
