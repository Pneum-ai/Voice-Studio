"""Typed in-memory job entities for asynchronous model transitions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class JobState(StrEnum):
  """Enumerate all externally visible job states."""

  QUEUED = "queued"
  RUNNING = "running"
  COMPLETE = "complete"
  CANCELLED = "cancelled"
  FAILED = "failed"


@dataclass(frozen=True, slots=True)
class Job:
  """Represent a retained snapshot of asynchronous work."""

  id: str
  kind: str
  resource_id: str
  state: JobState
  progress: float
  stage: str
  detail: str
  error: str | None
  created_at: datetime
  updated_at: datetime

  @property
  def terminal(self) -> bool:
    """Return whether no further state updates are expected."""
    return self.state in {
      JobState.COMPLETE,
      JobState.CANCELLED,
      JobState.FAILED,
    }
