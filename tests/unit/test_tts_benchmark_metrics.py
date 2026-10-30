"""Tests for Nari-compatible Voice Open streaming benchmark measurements."""

from __future__ import annotations

import math

import numpy as np

from scripts.benchmark_tts_streaming import Measurement, _poisson_schedule, _summary
from scripts.tts_benchmark_metrics import (
  AudibleOnsetDetector,
  AudioChunkObservation,
  distribution,
  evaluate_playback,
)


def _sine_pcm(duration_ms: int, *, amplitude: float = 0.1) -> bytes:
  """Create deterministic 24 kHz mono PCM16 with no DC offset."""
  samples = round(24_000 * duration_ms / 1000)
  positions = np.arange(samples, dtype=np.float32)
  waveform = amplitude * np.sin(2 * math.pi * 440 * positions / 24_000)
  return (waveform * 32767).astype("<i2").tobytes()


def test_audible_detector_requires_sustained_signal() -> None:
  """Match Nari's two-frame -45 dBFS onset contract across chunks."""
  detector = AudibleOnsetDetector()
  dc_samples = np.full(24_000 * 50 // 1000, 2_000, dtype="<i2").tobytes()
  silence = bytes(24_000 * 20 // 1000 * 2)
  audible = _sine_pcm(40)

  assert detector.feed(dc_samples) is None
  assert detector.feed(silence) is None
  assert detector.feed(audible[: len(audible) // 4]) is None
  assert detector.feed(audible[len(audible) // 4 :]) == 60.0


def test_playback_reports_transport_gap_after_buffer_exhaustion() -> None:
  """Count a late chunk as an underrun and map audible media onto wall time."""
  result = evaluate_playback(
    [
      AudioChunkObservation(arrival_ms=100, duration_ms=80),
      AudioChunkObservation(arrival_ms=220, duration_ms=80),
    ],
    audible_offset_ms=20,
  )

  assert result.first_playable_ms == 100
  assert result.audible_ttfa_ms == 120
  assert result.underruns == 1
  assert result.total_stalled_ms == 40
  assert result.audio_seconds == 0.16


def test_distribution_uses_nearest_rank_percentiles() -> None:
  """Keep percentile output equivalent to Nari's published benchmark method."""
  result = distribution([float(value) for value in range(1, 21)] + [None])

  assert result["p50"] == 10
  assert result["p95"] == 19
  assert result["p99"] == 20
  assert result["missing"] == 1


def test_poisson_schedule_is_deterministic_and_bounded() -> None:
  """Produce reproducible open-loop arrivals for cross-card comparisons."""
  first = _poisson_schedule(4, 2, 17)
  second = _poisson_schedule(4, 2, 17)

  assert first == second
  assert first == sorted(first)
  assert all(0 < value < 2 for value in first)


def test_summary_groups_stable_failure_codes() -> None:
  """Separate intentional overload shedding from transport and engine failures."""
  failed = Measurement(
    request_id="failed",
    scheduled_ms=0,
    dispatch_ms=0,
    scheduling_lag_ms=0,
    started_ms=1,
    completed_ms=None,
    playback=None,
    chunks=0,
    native_streaming=True,
    diagnostics=None,
    error_code="overloaded",
    error="RuntimeError: busy",
  )

  result = _summary(
    [failed],
    requested_rps=8,
    actual_rps=8,
    peak_in_flight=1,
  )

  assert result["failure_codes"] == {"overloaded": 1}
