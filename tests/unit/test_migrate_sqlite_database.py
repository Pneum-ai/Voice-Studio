"""Tests for safe SQLite migration into Linux-native WSL storage."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from scripts.migrate_sqlite_database import backup_database


def test_backup_database_copies_a_consistent_database(tmp_path: Path) -> None:
  """Copy committed rows and leave the source database unchanged."""
  source = tmp_path / "windows" / "voice.db"
  destination = tmp_path / "linux" / "voice.db"
  source.parent.mkdir()
  with sqlite3.connect(source) as connection:
    connection.execute("CREATE TABLE profiles (id TEXT PRIMARY KEY)")
    connection.execute("INSERT INTO profiles VALUES ('tyler')")

  backup_database(source, destination)

  with sqlite3.connect(destination) as connection:
    assert connection.execute("SELECT id FROM profiles").fetchall() == [("tyler",)]
  assert source.is_file()


def test_backup_database_rejects_the_same_path(tmp_path: Path) -> None:
  """Avoid replacing a database with itself when paths are misconfigured."""
  source = tmp_path / "voice.db"
  source.touch()

  with pytest.raises(ValueError, match="must differ"):
    backup_database(source, source)


def test_backup_database_rejects_uncheckpointed_wal(tmp_path: Path) -> None:
  """Require the old server to stop before migrating outstanding WAL data."""
  source = tmp_path / "voice.db"
  source.touch()
  source.with_name("voice.db-wal").write_bytes(b"pending")

  with pytest.raises(RuntimeError, match="active WAL"):
    backup_database(source, tmp_path / "native" / "voice.db")
