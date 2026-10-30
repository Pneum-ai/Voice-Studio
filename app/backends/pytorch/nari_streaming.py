"""Bounded asynchronous delivery and playback shaping for Nari speech."""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from app.core.errors import InferenceTimeoutError, OverloadedError
from app.utils.silence import LeadingSilenceTrimmer

LOGGER = logging.getLogger(__name__)
SUPERVISOR_RESTART_EXIT_CODE = 75


@dataclass(frozen=True, slots=True)
class NariRuntimeConfig:
  """Configure safe Nari admission, deadlines, trimming, and recovery."""

  execution_profile: Literal["auto", "rtx_24gb", "ttfa"] = "auto"
  policy_mode: Literal["safe", "candidate"] = "safe"
  max_active_0_6b: int = 8
  max_active_1_7b: int = 4
  blackwell_fp8: Literal["off", "auto", "force"] = "auto"
  capture_max_batch: int | None = None
  max_queued: int = 8
  queue_timeout_ms: int = 5_000
  first_audio_timeout_seconds: float = 3.0
  request_timeout_seconds: float = 60.0
  silence_trim: bool = True
  silence_threshold_dbfs: float = -45.0
  silence_max_ms: int = 3_000
  silence_preroll_ms: int = 10
  silence_min_output_ms: int = 160
  chunk_schedule: tuple[int, ...] = (4, 4, 8, 12)
  low_latency_chunk_schedule: tuple[int, ...] = (1, 2, 4, 8, 12)
  low_latency_max_active: int = 1
  adaptive_target_ms: int = 500
  watchdog_stall_seconds: float = 90.0
  exit_on_failure: bool = False

  def active_limit(self, model_id: str | None) -> int:
    """Return the model-aware concurrency ceiling."""
    return (
      self.max_active_1_7b
      if model_id in {"qwen-tts-1.7B", "qwen-tts-1.7B-customvoice"}
      else self.max_active_0_6b
    )

  def chunk_schedule_for(self, *, low_latency: bool) -> tuple[int, ...]:
    """Return the startup-reserved or loaded-card Codec schedule."""
    if low_latency:
      return self.low_latency_chunk_schedule
    return self.chunk_schedule


@dataclass(frozen=True, slots=True)
class NariRuntimeStatus:
  """Describe runtime readiness, admission pressure, and fatal state."""

  ready: bool
  reason: str
  active: int
  queued: int
  active_limit: int
  active_ceiling: int
  failure_id: str | None = None
  failure_detail: str | None = None
  policy_name: str | None = None
  capture_profile: str | None = None
  capture_profile_sha256: str | None = None
  capture_fallbacks: int = 0


@dataclass(slots=True)
class AdmissionLease:
  """Identify one admitted request and its exclusive low-latency startup reservation."""

  startup_reserved: bool
  startup_released: bool = False


class NariWatchdog:
  """Publish engine readiness and terminate a supervised child after fatal failure."""

  def __init__(
    self,
    engine: Any,
    admission: AdaptiveAdmission,
    config: NariRuntimeConfig,
  ) -> None:
    """Prepare a low-frequency engine health and progress monitor."""
    self._engine = engine
    self._admission = admission
    self._config = config
    self._stop = threading.Event()
    self._failure_id: str | None = None
    self._failure_detail: str | None = None
    self._tripped = False
    self._thread = threading.Thread(
      target=self._run,
      name="voice-qwen-nari-watchdog",
      daemon=True,
    )

  def start(self) -> None:
    """Start monitoring after the ordinary synthesis readiness proof passes."""
    self._thread.start()

  def stop(self) -> None:
    """Stop monitoring without blocking process shutdown indefinitely."""
    self._stop.set()
    if self._thread.is_alive() and self._thread is not threading.current_thread():
      self._thread.join(timeout=2)

  def status(self) -> NariRuntimeStatus:
    """Return a stable runtime snapshot for readiness and diagnostics APIs."""
    active, queued, limit, ceiling = self._admission.snapshot
    readiness = self._engine.readiness()
    ready = readiness.ready and self._failure_id is None
    return NariRuntimeStatus(
      ready=ready,
      reason=readiness.reason if self._failure_id is None else "watchdog_failed",
      active=active,
      queued=queued,
      active_limit=limit,
      active_ceiling=ceiling,
      failure_id=self._failure_id,
      failure_detail=self._failure_detail,
    )

  def _trip(self, detail: str) -> None:
    """Open the local circuit and request a clean process-level GPU reset."""
    if self._tripped:
      return
    self._tripped = True
    self._failure_id = self._failure_id or uuid.uuid4().hex[:12]
    self._failure_detail = self._failure_detail or detail
    LOGGER.critical(
      "Nari watchdog opened the circuit (failure_id=%s): %s",
      self._failure_id,
      detail,
    )
    if self._config.exit_on_failure:
      time.sleep(1)
      os._exit(SUPERVISOR_RESTART_EXIT_CODE)

  def _run(self) -> None:
    """Detect terminal phase changes and active engines that stop progressing."""
    previous: str | None = None
    was_active = False
    last_progress = time.monotonic()
    while not self._stop.wait(0.5):
      readiness = self._engine.readiness()
      active = self._engine.metrics().active
      fingerprint = repr(readiness.health)
      now = time.monotonic()
      # Idle health is unchanged; newly admitted work must get a full deadline
      # before the engine publishes its first stage-progress snapshot.
      if not active or not was_active or fingerprint != previous:
        last_progress = now
      previous = fingerprint
      was_active = bool(active)
      if getattr(readiness.phase, "value", readiness.phase) == "failed":
        failure_id, detail = self._engine.failure_snapshot()
        if failure_id is not None:
          self._failure_id = failure_id
          self._failure_detail = detail
        self._trip(detail or readiness.reason)
        return
      if active and now - last_progress >= self._config.watchdog_stall_seconds:
        self._trip(
          f"engine made no observable stage progress for "
          f"{self._config.watchdog_stall_seconds:g} seconds"
        )
        return


class AdaptiveAdmission:
  """Bound active Nari requests and adapt below a model-specific ceiling."""

  def __init__(self, config: NariRuntimeConfig, *, model_id: str | None) -> None:
    """Initialize counters and the recent first-audio latency window."""
    self._config = config
    self._ceiling = config.active_limit(model_id)
    self._limit = self._ceiling
    self._active = 0
    self._queued = 0
    self._startup_reserved = False
    self._condition = asyncio.Condition()
    self._latencies_ms: deque[float] = deque(maxlen=20)

  @property
  def snapshot(self) -> tuple[int, int, int, int]:
    """Return active, queued, effective-limit, and ceiling counters."""
    return self._active, self._queued, self._limit, self._ceiling

  async def enter(self) -> AdmissionLease:
    """Acquire one active slot and reserve an idle card for a short first chunk."""
    async with self._condition:

      def available() -> bool:
        """Return whether capacity is free and startup work has enough audio."""
        return self._active < self._limit and not self._startup_reserved

      if not available():
        if self._queued >= self._config.max_queued:
          raise OverloadedError(self._config.queue_timeout_ms)
        self._queued += 1
        try:
          async with asyncio.timeout(self._config.queue_timeout_ms / 1000):
            await self._condition.wait_for(available)
        except TimeoutError as error:
          raise OverloadedError(self._config.queue_timeout_ms) from error
        finally:
          self._queued -= 1
      startup_reserved = self._active == 0 and self._config.low_latency_max_active > 0
      self._active += 1
      self._startup_reserved = startup_reserved
      return AdmissionLease(startup_reserved=startup_reserved)

  async def release_startup(self, lease: AdmissionLease) -> None:
    """Admit waiting work after the startup stream has accumulated playback headroom."""
    async with self._condition:
      if not lease.startup_reserved or lease.startup_released:
        return
      lease.startup_released = True
      self._startup_reserved = False
      self._condition.notify_all()

  async def leave(self, lease: AdmissionLease | None = None) -> None:
    """Release one active slot and wake one event-loop waiter."""
    async with self._condition:
      if lease is not None and lease.startup_reserved and not lease.startup_released:
        lease.startup_released = True
        self._startup_reserved = False
      self._active -= 1
      if self._active < 0:
        raise RuntimeError("Nari active request count underflowed")
      self._condition.notify_all()

  async def observe(self, first_audio_ms: float | None, *, failed: bool) -> None:
    """Reduce sustained tail latency and cautiously restore recovered capacity."""
    async with self._condition:
      if failed:
        self._limit = max(1, self._limit - 1)
        self._latencies_ms.clear()
        return
      if first_audio_ms is None:
        return
      self._latencies_ms.append(first_audio_ms)
      if len(self._latencies_ms) < 8:
        return
      ordered = sorted(self._latencies_ms)
      p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
      if p95 > self._config.adaptive_target_ms and self._limit > 1:
        self._limit -= 1
        self._latencies_ms.clear()
      elif (
        len(self._latencies_ms) == self._latencies_ms.maxlen
        and p95 < self._config.adaptive_target_ms / 2
        and self._limit < self._ceiling
      ):
        self._limit += 1
        self._latencies_ms.clear()
        self._condition.notify(1)


@dataclass(frozen=True, slots=True)
class _BridgeFailure:
  """Carry one blocking PCM failure into the owning event loop."""

  error: BaseException


class AsyncPCMBridge:
  """Convert one blocking PCM stream into a bounded asynchronous iterator."""

  def __init__(self, stream: Any, *, queue_size: int = 4) -> None:
    """Start one bounded daemon reader without using asyncio's shared executor."""
    self._stream = stream
    self._loop = asyncio.get_running_loop()
    self._queue: asyncio.Queue[bytes | None | _BridgeFailure] = asyncio.Queue(queue_size)
    self._closed = threading.Event()
    self._publication_lock = threading.Lock()
    self._publication: Any | None = None
    self._thread = threading.Thread(
      target=self._pump,
      name="voice-qwen-nari-pcm",
      daemon=True,
    )
    self._thread.start()

  @property
  def timing(self) -> Any | None:
    """Return immutable engine timing metadata after stream retirement."""
    return self._stream.timing

  def _publish(self, item: bytes | None | _BridgeFailure) -> bool:
    """Apply event-loop queue backpressure from the blocking reader thread."""
    operation = self._queue.put(item)
    try:
      future = asyncio.run_coroutine_threadsafe(operation, self._loop)
    except BaseException:
      operation.close()
      return False
    try:
      with self._publication_lock:
        self._publication = future
        if self._closed.is_set():
          future.cancel()
      future.result()
    except BaseException:
      return False
    finally:
      with self._publication_lock:
        self._publication = None
    return True

  def _pump(self) -> None:
    """Read PCM until terminal state and publish exactly one terminal item."""
    try:
      while not self._closed.is_set():
        value = self._stream.read(timeout_s=None)
        if not self._publish(value) or value is None:
          return
    except BaseException as error:
      self._publish(_BridgeFailure(error))

  async def read(self, *, timeout_s: float) -> bytes | None:
    """Return the next PCM item or raise its original blocking-stream error."""
    try:
      async with asyncio.timeout(timeout_s):
        item = await self._queue.get()
    except TimeoutError as error:
      raise InferenceTimeoutError("Nari audio delivery exceeded its deadline.") from error
    if isinstance(item, _BridgeFailure):
      raise item.error
    return item

  def close(self) -> None:
    """Prevent further queue publication while engine cancellation wakes the reader."""
    self._closed.set()
    with self._publication_lock:
      if self._publication is not None:
        self._publication.cancel()


async def iter_with_deadline(
  bridge: AsyncPCMBridge,
  *,
  first_audio_timeout_s: float,
  request_timeout_s: float,
) -> AsyncIterator[bytes]:
  """Yield PCM while enforcing separate first-audio and whole-request deadlines."""
  started = time.monotonic()
  first = True
  while True:
    elapsed = time.monotonic() - started
    remaining = request_timeout_s - elapsed
    if remaining <= 0:
      raise InferenceTimeoutError("Nari generation exceeded its request deadline.")
    timeout = min(remaining, first_audio_timeout_s if first else remaining)
    value = await bridge.read(timeout_s=timeout)
    if value is None:
      return
    first = False
    yield value


__all__ = [
  "AdmissionLease",
  "AdaptiveAdmission",
  "AsyncPCMBridge",
  "LeadingSilenceTrimmer",
  "NariRuntimeConfig",
  "NariRuntimeStatus",
  "iter_with_deadline",
]
