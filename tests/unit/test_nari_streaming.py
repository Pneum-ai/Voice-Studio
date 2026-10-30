"""Bounded Nari admission, PCM bridging, and silence-gate tests."""

from __future__ import annotations

import asyncio
import threading
from collections import deque

import numpy as np
import pytest

from app.backends.pytorch.nari_streaming import (
  AdaptiveAdmission,
  AsyncPCMBridge,
  LeadingSilenceTrimmer,
  NariRuntimeConfig,
)
from app.core.errors import InferenceTimeoutError, OverloadedError
from app.utils.audio import AudioData


class _BlockingStream:
  """Return predetermined PCM items through the blocking Nari stream shape."""

  def __init__(self, values: list[bytes | None | BaseException]) -> None:
    """Store ordered values, one terminal item, or a raised failure."""
    self._values = deque(values)

  def read(self, *, timeout_s: float | None) -> bytes | None:
    """Return or raise the next configured stream item."""
    _ = timeout_s
    value = self._values.popleft()
    if isinstance(value, BaseException):
      raise value
    return value


def _tone(milliseconds: int, *, sample_rate: int = 24_000) -> np.ndarray:
  """Create deterministic speech-like energy above the audible threshold."""
  count = sample_rate * milliseconds // 1000
  phase = np.arange(count, dtype=np.float32) / sample_rate
  return (0.2 * np.sin(2 * np.pi * 440 * phase)).astype(np.float32)


def test_leading_silence_gate_keeps_preroll_and_opens_once() -> None:
  """Trim initial silence while preserving ten milliseconds before speech."""
  silence = np.zeros(24_000 * 150 // 1000, dtype=np.float32)
  samples = np.concatenate((silence, _tone(300)))
  trimmer = LeadingSilenceTrimmer(
    sample_rate=24_000,
    threshold_dbfs=-45,
    max_ms=3_000,
    preroll_ms=10,
  )

  output = trimmer.process(AudioData(samples, 24_000))
  followup = AudioData(_tone(50), 24_000)

  assert output is not None
  # A 20 ms window beginning at 140 ms already contains 10 ms of tone; with
  # 10 ms pre-roll the gate intentionally retains audio from 130 ms onward.
  assert output.samples.size == samples.size - 24_000 * 130 // 1000
  assert trimmer.process(followup) is followup
  assert trimmer.finish() is None


def test_leading_silence_gate_preserves_quiet_audio_after_safety_limit() -> None:
  """Avoid deleting a genuinely quiet voice when onset detection is uncertain."""
  quiet = np.full(24_000 * 120 // 1000, 0.0001, dtype=np.float32)
  trimmer = LeadingSilenceTrimmer(
    sample_rate=24_000,
    threshold_dbfs=-45,
    max_ms=100,
    preroll_ms=10,
  )

  output = trimmer.process(AudioData(quiet, 24_000))

  assert output is not None
  assert np.array_equal(output.samples, quiet)


def test_leading_silence_gate_waits_for_minimum_playback_headroom() -> None:
  """Avoid opening on a tiny speech tail that would immediately underrun."""
  silence = np.zeros(24_000 * 150 // 1000, dtype=np.float32)
  trimmer = LeadingSilenceTrimmer(
    sample_rate=24_000,
    threshold_dbfs=-45,
    max_ms=3_000,
    preroll_ms=10,
    min_output_ms=160,
  )

  assert trimmer.process(AudioData(np.concatenate((silence, _tone(30))), 24_000)) is None
  output = trimmer.process(AudioData(_tone(150), 24_000))

  assert output is not None
  assert output.samples.size / output.sample_rate >= 0.16


@pytest.mark.asyncio
async def test_admission_rejects_excess_queue_and_wakes_one_waiter() -> None:
  """Keep overload outside Nari and avoid an unbounded in-engine backlog."""
  controller = AdaptiveAdmission(
    NariRuntimeConfig(
      max_active_0_6b=2,
      max_queued=1,
      queue_timeout_ms=100,
    ),
    model_id="qwen-tts-0.6B",
  )
  first = await controller.enter()
  await controller.release_startup(first)
  await controller.enter()
  waiter = asyncio.create_task(controller.enter())
  await asyncio.sleep(0)

  with pytest.raises(OverloadedError) as caught:
    await controller.enter()
  assert caught.value.retry_after_ms == 100
  assert controller.snapshot == (2, 1, 2, 2)

  await controller.leave()
  await asyncio.wait_for(waiter, timeout=1)
  assert controller.snapshot == (2, 0, 2, 2)
  await controller.leave()
  await controller.leave()


@pytest.mark.asyncio
async def test_startup_reservation_holds_later_arrivals_until_headroom() -> None:
  """Keep a new request off the GPU until the idle-card stream releases its reservation."""
  controller = AdaptiveAdmission(
    NariRuntimeConfig(max_active_0_6b=2, max_queued=1, queue_timeout_ms=500),
    model_id="qwen-tts-0.6B",
  )
  first = await controller.enter()
  assert first.startup_reserved is True
  waiter = asyncio.create_task(controller.enter())
  await asyncio.sleep(0)
  assert controller.snapshot == (1, 1, 2, 2)

  await controller.release_startup(first)
  second = await asyncio.wait_for(waiter, timeout=1)

  assert second.startup_reserved is False
  assert controller.snapshot == (2, 0, 2, 2)
  await controller.leave(first)
  await controller.leave(second)


@pytest.mark.asyncio
async def test_admission_reduces_limit_after_sustained_tail_latency() -> None:
  """Lower effective concurrency when the recent p95 misses its target."""
  controller = AdaptiveAdmission(
    NariRuntimeConfig(max_active_0_6b=4, adaptive_target_ms=100),
    model_id="qwen-tts-0.6B",
  )
  for _ in range(8):
    await controller.observe(250, failed=False)

  assert controller.snapshot == (0, 0, 3, 4)


@pytest.mark.asyncio
async def test_pcm_bridge_delivers_terminal_without_shared_executor_polling() -> None:
  """Move blocking PCM reads to one request-owned daemon bridge."""
  bridge = AsyncPCMBridge(_BlockingStream([b"\x01\x00", b"\x02\x00", None]))
  try:
    assert await bridge.read(timeout_s=1) == b"\x01\x00"
    assert await bridge.read(timeout_s=1) == b"\x02\x00"
    assert await bridge.read(timeout_s=1) is None
  finally:
    bridge.close()


@pytest.mark.asyncio
async def test_pcm_bridge_preserves_failure_and_has_a_deadline() -> None:
  """Surface the original reader failure and convert stalls into typed timeouts."""
  failure = RuntimeError("original engine failure")
  failed = AsyncPCMBridge(_BlockingStream([failure]))
  try:
    with pytest.raises(RuntimeError, match="original engine failure"):
      await failed.read(timeout_s=1)
  finally:
    failed.close()

  class NeverReturns:
    """Block longer than the bridge deadline."""

    def read(self, *, timeout_s: float | None) -> bytes | None:
      """Wait on an event that the test never sets."""
      _ = timeout_s
      threading.Event().wait(10)
      return None

  stalled = AsyncPCMBridge(NeverReturns())
  try:
    with pytest.raises(InferenceTimeoutError):
      await stalled.read(timeout_s=0.01)
  finally:
    stalled.close()
