"""Safely copy a Voice Studio SQLite database to Linux-native WSL storage."""

from __future__ import annotations

import argparse
import sqlite3
from contextlib import closing
from pathlib import Path


def backup_database(source: Path, destination: Path) -> None:
  """Create a consistent SQLite backup without copying transient WAL sidecars.

  Args:
    source: Existing Voice Studio database to read.
    destination: New database path to create atomically.

  Raises:
    FileNotFoundError: If the source database does not exist.
    ValueError: If source and destination resolve to the same path.
    RuntimeError: If the source has uncheckpointed WAL data from a running server.
    sqlite3.Error: If SQLite cannot read, validate, or back up the database.
  """
  source = source.resolve()
  destination = destination.resolve()
  if not source.is_file():
    raise FileNotFoundError(f"Source database does not exist: {source}")
  if source == destination:
    raise ValueError("Source and destination database paths must differ.")
  source_wal = source.with_name(f"{source.name}-wal")
  if source_wal.is_file() and source_wal.stat().st_size > 0:
    raise RuntimeError(
      "Source database has an active WAL file; stop the old server before migrating."
    )

  destination.parent.mkdir(parents=True, exist_ok=True)
  temporary = destination.with_name(f".{destination.name}.migrating")
  if temporary.exists():
    temporary.unlink()

  # immutable=1 prevents SQLite from trying to create WAL bookkeeping files on /mnt/c.
  source_uri = f"file:{source.as_posix()}?mode=ro&immutable=1"
  try:
    with closing(sqlite3.connect(source_uri, uri=True)) as source_connection:
      integrity = source_connection.execute("PRAGMA integrity_check").fetchone()
      if integrity != ("ok",):
        raise sqlite3.DatabaseError(f"Source integrity check failed: {integrity}")
      with closing(sqlite3.connect(temporary)) as destination_connection:
        source_connection.backup(destination_connection)
        copied_integrity = destination_connection.execute("PRAGMA integrity_check").fetchone()
        if copied_integrity != ("ok",):
          raise sqlite3.DatabaseError(f"Destination integrity check failed: {copied_integrity}")
    temporary.replace(destination)
  except BaseException:
    temporary.unlink(missing_ok=True)
    raise


def parse_args() -> argparse.Namespace:
  """Parse source and destination database paths from the command line."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("source", type=Path)
  parser.add_argument("destination", type=Path)
  return parser.parse_args()


def main() -> None:
  """Back up the requested database and report the completed destination."""
  arguments = parse_args()
  backup_database(arguments.source, arguments.destination)
  print(f"SQLite database ready at {arguments.destination.resolve()}")


if __name__ == "__main__":
  main()
