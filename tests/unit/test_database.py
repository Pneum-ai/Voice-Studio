"""Fresh SQLite schema and incompatible-database behavior."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text

from app.persistence.database import Database


@pytest.mark.asyncio
async def test_database_uses_versioned_pragmas_and_schema(tmp_path: Path) -> None:
  """Initialize the new schema with its deliberate SQLite performance settings."""
  database = Database(tmp_path / "voice.db")
  try:
    await database.initialize()
    async with database.engine.connect() as connection:
      version = await connection.scalar(
        text("SELECT value FROM schema_metadata WHERE key = 'schema_version'")
      )
      foreign_keys = await connection.scalar(text("PRAGMA foreign_keys"))
      journal_mode = await connection.scalar(text("PRAGMA journal_mode"))
      synchronous = await connection.scalar(text("PRAGMA synchronous"))
      sample_columns = await connection.execute(text("PRAGMA table_info(profile_samples)"))
    assert version == "5"
    assert "style_controls_json" in {str(row[1]) for row in sample_columns}
    assert foreign_keys == 1
    assert str(journal_mode).casefold() == "wal"
    assert synchronous == 1
  finally:
    await database.close()


@pytest.mark.asyncio
async def test_database_rejects_unversioned_existing_schema(tmp_path: Path) -> None:
  """Refuse an unrelated or legacy database instead of migrating it."""
  path = tmp_path / "voice.db"
  with sqlite3.connect(path) as connection:
    connection.execute("CREATE TABLE voice_profiles (id TEXT PRIMARY KEY)")
  database = Database(path)
  try:
    with pytest.raises(RuntimeError, match="choose a new data directory"):
      await database.initialize()
  finally:
    await database.close()


@pytest.mark.asyncio
async def test_database_applies_later_ordered_migration(tmp_path: Path) -> None:
  """Upgrade a prior supported version without rebuilding relational data."""
  path = tmp_path / "voice.db"
  database = Database(path)
  await database.initialize()
  await database.close()
  with sqlite3.connect(path) as connection:
    connection.execute("DROP TRIGGER generations_fts_insert")
    connection.execute("DROP TRIGGER generations_fts_delete")
    connection.execute("DROP TRIGGER generations_fts_update")
    connection.execute("DROP TABLE generations_fts")
    connection.execute("DROP INDEX ix_generations_model_created")
    connection.execute("UPDATE schema_metadata SET value = '1' WHERE key = 'schema_version'")

  upgraded = Database(path)
  try:
    await upgraded.initialize()
    async with upgraded.engine.connect() as connection:
      version = await connection.scalar(
        text("SELECT value FROM schema_metadata WHERE key = 'schema_version'")
      )
      search_table = await connection.scalar(
        text("SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'generations_fts'")
      )
    assert version == "5"
    assert search_table == "generations_fts"
  finally:
    await upgraded.close()


@pytest.mark.asyncio
async def test_database_adds_sample_style_controls_migration(tmp_path: Path) -> None:
  """Upgrade version-four sample rows with empty optional style metadata."""
  path = tmp_path / "voice.db"
  database = Database(path)
  await database.initialize()
  await database.close()
  with sqlite3.connect(path) as connection:
    connection.execute("ALTER TABLE profile_samples DROP COLUMN style_controls_json")
    connection.execute("UPDATE schema_metadata SET value = '4' WHERE key = 'schema_version'")

  upgraded = Database(path)
  try:
    await upgraded.initialize()
    async with upgraded.engine.connect() as connection:
      version = await connection.scalar(
        text("SELECT value FROM schema_metadata WHERE key = 'schema_version'")
      )
      sample_columns = await connection.execute(text("PRAGMA table_info(profile_samples)"))
    assert version == "5"
    assert "style_controls_json" in {str(row[1]) for row in sample_columns}
  finally:
    await upgraded.close()
