"""Bridge optional vendored Nari engine timings into request diagnostics."""

from __future__ import annotations

from typing import Any

from app.services.speech_diagnostics import record_diagnostics


def record_nari_diagnostics(timing: Any | None) -> None:
  """Publish a timing snapshot when the WebSocket requested diagnostics."""
  if timing is not None:
    record_diagnostics(timing.as_dict())


__all__ = ["record_nari_diagnostics"]
