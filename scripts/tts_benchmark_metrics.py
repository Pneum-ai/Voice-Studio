"""Nari-compatible audible-onset, playback, and percentile benchmark helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

SAMPLE_RATE = 24_000


@dataclass(frozen=True, slots=True)
class AudioChunkObservation:
  """Record one PCM chunk's client arrival time and playable duration."""

  arrival_ms: float
  duration_ms: float


@dataclass(frozen=True, slots=True)
class PlaybackResult:
  """Describe immediate playback timing and transport stalls."""

  first_playable_ms: float | None
  audible_ttfa_ms: float | None
  leading_silence_ms: float | None
  audio_seconds: float
  underruns: int
  total_stalled_ms: float
  largest_stall_ms: float


class AudibleOnsetDetector:
  """Match Nari's sustained, DC-removed -45 dBFS speech-onset detector."""

  frame_ms = 20
  hop_ms = 10
  threshold_dbfs = -45.0
  consecutive_active_frames = 2

  def __init__(self) -> None:
    """Initialize an empty mono PCM16 stream detector."""
    self._pcm = bytearray()
    self._next_window_start = 0
    self._active_run_start: int | None = None
    self._active_run_length = 0
    self.onset_ms: float | None = None

  def feed(self, pcm: bytes) -> float | None:
    """Consume whole mono PCM16 frames and return sustained onset when known."""
    if self.onset_ms is not None:
      return self.onset_ms
    if len(pcm) % 2:
      raise ValueError("PCM16 chunks must contain complete samples")
    self._pcm.extend(pcm)
    frame_samples = round(SAMPLE_RATE * self.frame_ms / 1000)
    hop_samples = round(SAMPLE_RATE * self.hop_ms / 1000)
    threshold = 10 ** (self.threshold_dbfs / 20)
    total_samples = len(self._pcm) // 2
    while self._next_window_start + frame_samples <= total_samples:
      start = self._next_window_start
      samples = np.frombuffer(
        self._pcm[start * 2 : (start + frame_samples) * 2],
        dtype="<i2",
      ).astype(np.float32)
      samples /= 32768.0
      samples -= float(np.mean(samples))
      rms = float(np.sqrt(np.mean(samples * samples)))
      if rms >= threshold:
        if self._active_run_length == 0:
          self._active_run_start = start
        self._active_run_length += 1
        if self._active_run_length >= self.consecutive_active_frames:
          assert self._active_run_start is not None
          self.onset_ms = self._active_run_start / SAMPLE_RATE * 1000
          return self.onset_ms
      else:
        self._active_run_start = None
        self._active_run_length = 0
      self._next_window_start += hop_samples
    return None


def evaluate_playback(
  chunks: list[AudioChunkObservation],
  *,
  audible_offset_ms: float | None,
) -> PlaybackResult:
  """Map media chunks onto an immediate-playback wall-clock timeline."""
  positive = [chunk for chunk in chunks if chunk.duration_ms > 0]
  if not positive:
    return PlaybackResult(None, None, None, 0.0, 0, 0.0, 0.0)
  first_playable = positive[0].arrival_ms
  playback_deadline = first_playable
  media_cursor = 0.0
  audible_ttfa: float | None = None
  stalls: list[float] = []
  for chunk in positive:
    if chunk.arrival_ms > playback_deadline:
      stalls.append(chunk.arrival_ms - playback_deadline)
      playback_deadline = chunk.arrival_ms
    if (
      audible_offset_ms is not None
      and audible_ttfa is None
      and media_cursor <= audible_offset_ms < media_cursor + chunk.duration_ms
    ):
      audible_ttfa = playback_deadline + audible_offset_ms - media_cursor
    playback_deadline += chunk.duration_ms
    media_cursor += chunk.duration_ms
  return PlaybackResult(
    first_playable_ms=first_playable,
    audible_ttfa_ms=audible_ttfa,
    leading_silence_ms=audible_offset_ms,
    audio_seconds=media_cursor / 1000,
    underruns=len(stalls),
    total_stalled_ms=sum(stalls),
    largest_stall_ms=max(stalls, default=0.0),
  )


def distribution(values: list[float | None]) -> dict[str, Any]:
  """Return Nari's nearest-rank p50/p95/p99 distribution and missing count."""
  ordered = sorted(value for value in values if value is not None and math.isfinite(value))
  if not ordered:
    return {"samples": 0, "missing": len(values)}

  def percentile(percent: int) -> float:
    """Select one nearest-rank percentile from the ordered samples."""
    index = max(0, math.ceil(percent * len(ordered) / 100) - 1)
    return ordered[index]

  return {
    "samples": len(ordered),
    "missing": len(values) - len(ordered),
    "min": ordered[0],
    "p50": percentile(50),
    "p95": percentile(95),
    "p99": percentile(99),
    "max": ordered[-1],
    "mean": sum(ordered) / len(ordered),
  }


def pcm_duration_ms(pcm: bytes) -> float:
  """Return mono 24 kHz PCM16 duration in milliseconds."""
  if len(pcm) % 2:
    raise ValueError("PCM16 chunks must contain complete samples")
  return len(pcm) / 2 / SAMPLE_RATE * 1000


__all__ = [
  "AudioChunkObservation",
  "AudibleOnsetDetector",
  "PlaybackResult",
  "distribution",
  "evaluate_playback",
  "pcm_duration_ms",
]
