"""Playback-aware admission for the continuous MLX scheduler.

The caller reports audio actually released after onset trimming. Decisions use that
playable duration, not generated token counts that could include discarded silence.
All time values are monotonic seconds on the GPU-owner's host.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal


class PlaybackPolicy:
  """Reserve enough playback time for prediction, decoding, and optional new prefill."""

  def __init__(self) -> None:
    """Start with conservative service estimates until real operations are timed."""
    self.ends: dict[int, float] = {}
    self.token_seconds = 0.025
    self.codec_seconds = 0.025
    self.prefill_per_row = 0.025
    self.prefill_seconds = 0.100
    self.margin = 0.100
    self.deferred_admissions = 0
    self.urgent_decodes = 0

  def received(self, identity: int, duration: float, now: float) -> None:
    """Track queued playable audio, restarting the timeline if playback already stalled."""
    if duration <= 0:
      return
    self.ends[identity] = max(now, self.ends.get(identity, now)) + duration

  def forget(self, identity: int) -> None:
    """Discard timing for a completed or cancelled request."""
    self.ends.pop(identity, None)

  def observe(
    self, operation: Literal["token", "codec", "prefill"], seconds: float, rows: int = 1
  ) -> None:
    """React immediately to slower work and decay old estimates gradually."""
    name = {"token": "token_seconds", "codec": "codec_seconds", "prefill": "prefill_per_row"}[
      operation
    ]
    cost = seconds / max(1, rows) if operation == "prefill" else seconds
    setattr(self, name, max(cost, getattr(self, name) * 0.9))
    if operation == "prefill":
      # Prefill has a fixed launch/cache cost; dividing a large batch by its rows
      # understates the cost of admitting a single newcomer.
      self.prefill_seconds = max(seconds, self.prefill_seconds * 0.9)

  def can_admit(self, active: Iterable[int], rows: int, now: float, until_decode: int) -> bool:
    """Allow prefill only after existing streams have sufficient playable headroom."""
    identities = list(active)
    if not identities:
      return True
    reserve = (
      max(self.prefill_seconds, self.prefill_per_row * rows)
      + self.token_seconds * until_decode
      + self.codec_seconds
      + self.margin
    )
    allowed = all(self.ends.get(identity, now) - now >= reserve for identity in identities)
    if not allowed:
      self.deferred_admissions += 1
    return allowed

  def decode_due(
    self, identity: int, pending: int, decoded: int, now: float, until_decode: int
  ) -> bool:
    """Start at four frames, retry startup in two-frame increments, and protect playback."""
    if pending <= 0:
      return False
    if identity not in self.ends:
      return pending >= (4 if decoded == 0 else 2)
    reserve = self.token_seconds * until_decode + self.codec_seconds + self.margin
    urgent = self.ends[identity] - now <= reserve
    if urgent:
      self.urgent_decodes += 1
    return urgent
