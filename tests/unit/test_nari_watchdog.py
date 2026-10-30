"""Exercise watchdog deadlines across idle, busy, and failed engine states."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from app.backends.pytorch.nari_streaming import (
  AdaptiveAdmission,
  NariRuntimeConfig,
  NariWatchdog,
)


@dataclass(frozen=True)
class _Sample:
  """Describe one health observation at a deterministic elapsed time."""

  seconds: float
  active: int
  health: int = 0
  phase: str = "ready"


class _EngineTimeline:
  """Advance engine observations without sleeping or starting GPU workers."""

  def __init__(self, samples: list[_Sample]) -> None:
    """Store the sequence and an initial idle observation."""
    self._samples = iter(samples)
    self.current = _Sample(0, 0)

  def wait(self, timeout: float) -> bool:
    """Advance a poll or stop the watchdog when observations are exhausted."""
    sample = next(self._samples, None)
    if sample is None:
      return True
    self.current = sample
    return False

  def readiness(self) -> SimpleNamespace:
    """Expose health and readiness for the current observation."""
    return SimpleNamespace(
      health=self.current.health,
      phase=self.current.phase,
      reason="ready",
      ready=True,
    )

  def metrics(self) -> SimpleNamespace:
    """Expose outstanding engine work at the current observation."""
    return SimpleNamespace(active=self.current.active)

  def failure_snapshot(self) -> tuple[str, str]:
    """Expose a terminal engine error independently of the stall deadline."""
    return "engine-failure", "CUDA worker failed"


@pytest.mark.parametrize(
  ("samples", "should_trip"),
  [
    ([_Sample(1, 0), _Sample(200, 0), _Sample(200.5, 1)], False),
    ([_Sample(1, 1), _Sample(90, 1), _Sample(91, 1)], True),
    ([_Sample(1, 0), _Sample(200, 1), _Sample(289, 1)], False),
    ([_Sample(1, 0), _Sample(200, 1), _Sample(290, 1)], True),
    ([_Sample(1, 1), _Sample(80, 1, 1), _Sample(160, 1, 1)], False),
    ([_Sample(1, 1), _Sample(80, 0), _Sample(200, 1)], False),
    ([_Sample(1, 0, phase="failed")], True),
  ],
  ids=["idle", "stalled", "fresh-work", "fresh-stall", "progress", "idle-gap", "failed"],
)
def test_watchdog_deadlines(samples: list[_Sample], should_trip: bool) -> None:
  """Only sustained active stalls or terminal failures may open the circuit."""
  timeline = _EngineTimeline(samples)
  config = NariRuntimeConfig()
  watchdog = NariWatchdog(timeline, AdaptiveAdmission(config, model_id=None), config)
  with (
    patch.object(watchdog, "_stop", timeline),
    patch(
      "app.backends.pytorch.nari_streaming.time.monotonic",
      side_effect=lambda: timeline.current.seconds,
    ),
  ):
    watchdog._run()
  assert watchdog.status().ready is not should_trip
  assert (watchdog.status().failure_id is not None) is should_trip
