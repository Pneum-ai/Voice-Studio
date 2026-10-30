"""Async SQLite engine lifecycle with performance and safety pragmas."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, cast

from sqlalchemy import event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
  AsyncEngine,
  AsyncSession,
  async_sessionmaker,
  create_async_engine,
)

from app.persistence.migrations import LATEST_SCHEMA_VERSION, MIGRATIONS
from app.persistence.orm import SchemaMetadataRecord


class Database:
  """Own the async engine and create short-lived database sessions."""

  def __init__(self, path: Path) -> None:
    """Configure a database without opening connections yet."""
    self.path = path
    self.engine: AsyncEngine = create_async_engine(
      f"sqlite+aiosqlite:///{path}",
      pool_pre_ping=True,
    )
    self.sessions = async_sessionmaker(
      self.engine,
      class_=AsyncSession,
      expire_on_commit=False,
    )
    self._configure_pragmas(self.engine.sync_engine)

  @staticmethod
  def _configure_pragmas(engine: Engine) -> None:
    """Install SQLite connection pragmas on the synchronous engine facade."""

    @event.listens_for(engine, "connect")
    def set_sqlite_pragmas(dbapi_connection: object, _: object) -> None:
      """Configure every pooled SQLite connection consistently."""
      cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
      cursor.execute("PRAGMA foreign_keys = ON")
      cursor.execute("PRAGMA journal_mode = WAL")
      cursor.execute("PRAGMA synchronous = NORMAL")
      cursor.execute("PRAGMA busy_timeout = 5000")
      cursor.close()

  async def initialize(self) -> None:
    """Apply ordered migrations without recognizing or importing legacy schemas."""
    self.path.parent.mkdir(parents=True, exist_ok=True)
    async with self.engine.begin() as connection:
      table_names = await connection.run_sync(
        lambda sync_connection: set(inspect(sync_connection).get_table_names())
      )
      if table_names and "schema_metadata" not in table_names:
        raise RuntimeError(
          "Database is not initialized for this service; choose a new data directory."
        )
      schema_table = cast(Any, SchemaMetadataRecord.__table__)
      await connection.run_sync(
        lambda sync_connection: schema_table.create(
          sync_connection,
          checkfirst=True,
        )
      )
      result = await connection.execute(
        text("SELECT value FROM schema_metadata WHERE key = 'schema_version'")
      )
      version = result.scalar_one_or_none()
      try:
        current_version = int(version) if version is not None else 0
      except ValueError as error:
        raise RuntimeError("Database schema version is invalid.") from error
      if current_version > LATEST_SCHEMA_VERSION:
        raise RuntimeError(f"Unsupported database schema version: {version}")
      for migration in MIGRATIONS:
        if migration.version <= current_version:
          continue
        await migration.apply(connection)
        await connection.execute(
          text(
            "INSERT INTO schema_metadata (key, value) "
            "VALUES ('schema_version', :version) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
          ),
          {"version": str(migration.version)},
        )

  async def session(self) -> AsyncIterator[AsyncSession]:
    """Yield a transaction-neutral session for dependency injection."""
    async with self.sessions() as session:
      yield session

  async def check_integrity(self) -> None:
    """Raise when SQLite reports integrity or foreign-key failures."""
    async with self.engine.connect() as connection:
      integrity = await connection.execute(text("PRAGMA integrity_check"))
      if integrity.scalar_one() != "ok":
        raise RuntimeError("SQLite integrity check failed.")
      foreign_keys = await connection.execute(text("PRAGMA foreign_key_check"))
      if foreign_keys.first() is not None:
        raise RuntimeError("SQLite foreign-key check failed.")

  async def close(self) -> None:
    """Dispose all pooled database connections."""
    await self.engine.dispose()
