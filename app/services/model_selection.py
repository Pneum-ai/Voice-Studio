"""Persist the selected TTS model across supervised CUDA process restarts."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.jobs import Job, JobState

MODEL_SWITCH_RESTART_EXIT_CODE = 76


@dataclass(frozen=True, slots=True)
class PendingModelSwitch:
  """Describe one model-switch job that must survive a worker restart."""

  job_id: str
  model_id: str
  created_at: datetime

  def as_job(self) -> Job:
    """Restore the public job snapshot exposed before the old worker exited."""
    return Job(
      id=self.job_id,
      kind="model_state",
      resource_id=self.model_id,
      state=JobState.RUNNING,
      progress=0.72,
      stage="restarting",
      detail="Restarting the CUDA worker to release model memory",
      error=None,
      created_at=self.created_at,
      updated_at=datetime.now(UTC),
    )


@dataclass(frozen=True, slots=True)
class ModelSelectionState:
  """Hold the durable TTS selection and an optional restart-spanning job."""

  selected_model: str | None
  pending: PendingModelSwitch | None


class ModelSelectionStore:
  """Read and atomically replace the small supervisor coordination file."""

  def __init__(self, path: Path) -> None:
    """Store the state-file path without touching the filesystem."""
    self.path = path
    self._lock = threading.Lock()

  def read(self) -> ModelSelectionState:
    """Return validated state, treating a missing file as no selection."""
    with self._lock:
      return self._read_unlocked()

  def select(self, model_id: str) -> None:
    """Persist a successfully loaded model and clear any completed switch."""
    self._write(ModelSelectionState(selected_model=model_id, pending=None))

  def prepare(self, job: Job, model_id: str) -> None:
    """Persist a target and job identity before terminating the CUDA worker."""
    pending = PendingModelSwitch(job.id, model_id, job.created_at)
    self._write(ModelSelectionState(selected_model=model_id, pending=pending))

  def complete(self, job_id: str) -> None:
    """Clear a matching pending marker while retaining the selected model."""
    with self._lock:
      current = self._read_unlocked()
      if current.pending is None or current.pending.job_id != job_id:
        return
      self._write_unlocked(ModelSelectionState(current.selected_model, None))

  def _read_unlocked(self) -> ModelSelectionState:
    """Decode state while the caller owns the process-local file lock."""
    try:
      payload = json.loads(self.path.read_text(encoding="utf-8"))
    except FileNotFoundError:
      return ModelSelectionState(None, None)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
      return ModelSelectionState(None, None)
    if not isinstance(payload, dict):
      return ModelSelectionState(None, None)
    selected = payload.get("selected_model")
    selected_model = selected if isinstance(selected, str) else None
    pending_payload = payload.get("pending")
    pending = self._decode_pending(pending_payload)
    return ModelSelectionState(selected_model, pending)

  @staticmethod
  def _decode_pending(payload: Any) -> PendingModelSwitch | None:
    """Decode a pending record without trusting manually edited state."""
    if not isinstance(payload, dict):
      return None
    try:
      job_id = str(payload["job_id"])
      model_id = str(payload["model_id"])
      created_at = datetime.fromisoformat(str(payload["created_at"]))
    except (KeyError, TypeError, ValueError):
      return None
    if created_at.tzinfo is None:
      created_at = created_at.replace(tzinfo=UTC)
    return PendingModelSwitch(job_id, model_id, created_at)

  def _write(self, state: ModelSelectionState) -> None:
    """Serialize one state update under the process-local lock."""
    with self._lock:
      self._write_unlocked(state)

  def _write_unlocked(self, state: ModelSelectionState) -> None:
    """Atomically replace state so a worker never observes partial JSON."""
    pending = None
    if state.pending is not None:
      pending = {
        "job_id": state.pending.job_id,
        "model_id": state.pending.model_id,
        "created_at": state.pending.created_at.isoformat(),
      }
    payload = {"selected_model": state.selected_model, "pending": pending}
    self.path.parent.mkdir(parents=True, exist_ok=True)
    temporary = self.path.with_suffix(f"{self.path.suffix}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, self.path)


def has_pending_model_switch(path: Path) -> bool:
  """Return whether a supervisor should retry a failed replacement worker."""
  return ModelSelectionStore(path).read().pending is not None
