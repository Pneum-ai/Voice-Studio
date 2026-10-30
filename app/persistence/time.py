"""Timestamp conversion helpers for compact SQLite integer storage."""

from __future__ import annotations

from datetime import UTC, datetime


def now_ms() -> int:
  """Return the current UTC Unix timestamp in milliseconds."""
  return int(datetime.now(UTC).timestamp() * 1000)


def datetime_from_ms(value: int) -> datetime:
  """Convert UTC Unix milliseconds to a timezone-aware datetime."""
  return datetime.fromtimestamp(value / 1000, tz=UTC)
