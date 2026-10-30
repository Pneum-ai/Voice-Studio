"""Dynamic batching so concurrent speech requests share one generation pass."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from app.backends.factory import BackendBundle
from app.backends.protocols import BatchGenerationRequest
from app.utils.audio import AudioData

LOGGER = logging.getLogger(__name__)
LENGTH_RATIO = 2.0


@dataclass(slots=True)
class _QueuedRequest:
  """Hold one queued generation and the future awaiting its waveform."""

  model_id: str
  request: BatchGenerationRequest
  seed: int | None
  cancel_event: asyncio.Event | None
  future: asyncio.Future[AudioData]

  @property
  def group_key(self) -> str:
    """Return the model identity of batches this request may join."""
    return self.model_id

  @property
  def runs_alone(self) -> bool:
    """Return whether this request must run by itself to honor its seed."""
    return self.seed is not None

  @property
  def abandoned(self) -> bool:
    """Return whether nothing is waiting for this request any more."""
    return self.future.done() or (self.cancel_event is not None and self.cancel_event.is_set())


class TTSBatcher:
  """Collect concurrent speech requests and run them as batched passes.

  Compatible backends may share GPU work across requests. Actual throughput and
  latency depend on the model and hardware. This scheduler drives complete-waveform
  generation while coordinating access with streaming requests.
  """

  def __init__(
    self,
    backends: BackendBundle,
    *,
    enabled: bool = True,
    max_batch_size: int = 8,
    window_seconds: float = 0.05,
  ) -> None:
    """Configure batch limits and start with an idle queue and no worker."""
    self.backends = backends
    self._max_batch_size = max(1, max_batch_size)
    self._window_seconds = max(0.0, window_seconds)
    self._enabled = enabled and self._max_batch_size > 1
    self._queue: asyncio.Queue[_QueuedRequest] = asyncio.Queue()
    self._carried: _QueuedRequest | None = None
    self._worker: asyncio.Task[None] | None = None
    self._generation_lock = asyncio.Lock()
    self._stream_condition = asyncio.Condition()
    self._active_stream_key: str | None = None
    self._active_stream_count = 0
    self._active_stream_ready = False
    self._managed: set[asyncio.Task[AudioData]] = set()

  async def submit(
    self,
    *,
    model_id: str,
    request: BatchGenerationRequest,
    seed: int | None,
    cancel_event: asyncio.Event | None = None,
  ) -> AudioData:
    """Queue one generation and wait for the waveform its batch produces."""
    if getattr(self.backends.tts, "manages_request_queue", False):
      task = asyncio.create_task(self._submit_managed(model_id, request, seed, cancel_event))
      self._managed.add(task)
      try:
        return await task
      finally:
        self._managed.discard(task)
    loop = asyncio.get_running_loop()
    queued = _QueuedRequest(
      model_id=model_id,
      request=request,
      seed=seed,
      cancel_event=cancel_event,
      future=loop.create_future(),
    )
    self._ensure_worker()
    await self._queue.put(queued)
    return await queued.future

  async def _submit_managed(
    self,
    model_id: str,
    request: BatchGenerationRequest,
    seed: int | None,
    cancel_event: asyncio.Event | None,
  ) -> AudioData:
    """Let a backend-owned scheduler combine saved and live speech under one model lease."""
    async with self.streaming_section(model_id):
      generated = await self.backends.tts.generate_batch(
        [request], seed=seed, cancel_events=[cancel_event]
      )
      if len(generated) != 1:
        raise RuntimeError("The backend did not return the requested waveform.")
      return generated[0]

  def _ensure_worker(self) -> None:
    """Start the drain worker on first use so no explicit startup is required."""
    if self._worker is None or self._worker.done():
      self._worker = asyncio.create_task(self._drain(), name="voice-tts-batcher")

  async def stop(self) -> None:
    """Cancel the worker and fail every request still waiting for a batch."""
    worker = self._worker
    self._worker = None
    managed = list(self._managed)
    for task in managed:
      task.cancel()
    await asyncio.gather(*managed, return_exceptions=True)
    if worker is not None and not worker.done():
      worker.cancel()
      await asyncio.gather(worker, return_exceptions=True)
    pending = [] if self._carried is None else [self._carried]
    self._carried = None
    while not self._queue.empty():
      pending.append(self._queue.get_nowait())
    for queued in pending:
      if not queued.future.done():
        queued.future.set_exception(RuntimeError("The speech batcher is shutting down."))

  @asynccontextmanager
  async def streaming_section(
    self,
    model_id: str,
  ) -> AsyncIterator[None]:
    """Hold the generation lock so a native stream cannot overlap a batch.

    Streaming backends bypass batching, so they take the same model lease the
    worker holds while a complete-waveform batch runs.
    """
    concurrent = bool(getattr(self.backends.tts, "supports_concurrent_streaming", False))
    if not concurrent:
      async with self._generation_lock:
        yield
      return
    await self._enter_concurrent_stream(model_id)
    try:
      yield
    finally:
      await self._leave_concurrent_stream(model_id)

  async def _enter_concurrent_stream(
    self,
    key: str,
  ) -> None:
    """Join compatible native streams while reserving the loaded model."""
    leader = False
    while True:
      async with self._stream_condition:
        if self._active_stream_key is None:
          self._active_stream_key = key
          self._active_stream_count = 1
          self._active_stream_ready = False
          leader = True
          break
        if self._active_stream_key == key and self._active_stream_ready:
          self._active_stream_count += 1
          return
        await self._stream_condition.wait()
    acquired = False
    try:
      await self._generation_lock.acquire()
      acquired = True
    except BaseException:
      if acquired:
        self._generation_lock.release()
      async with self._stream_condition:
        if leader and self._active_stream_key == key:
          self._active_stream_key = None
          self._active_stream_count = 0
          self._active_stream_ready = False
          self._stream_condition.notify_all()
      raise
    async with self._stream_condition:
      self._active_stream_ready = True
      self._stream_condition.notify_all()

  async def _leave_concurrent_stream(self, key: str) -> None:
    """Release one shared stream lease and permit a model swap last."""
    async with self._stream_condition:
      if self._active_stream_key != key or self._active_stream_count < 1:
        raise RuntimeError("Native stream lease state is inconsistent.")
      self._active_stream_count -= 1
      if self._active_stream_count == 0:
        self._active_stream_key = None
        self._active_stream_ready = False
        self._generation_lock.release()
        self._stream_condition.notify_all()

  async def _drain(self) -> None:
    """Continuously assemble and run batches until the worker is cancelled."""
    while True:
      try:
        batch = await self._collect()
        if batch:
          await self._run_batch(batch)
      except asyncio.CancelledError:
        raise
      except Exception as error:
        LOGGER.exception("Speech batcher iteration failed", exc_info=error)

  async def _collect(self) -> list[_QueuedRequest]:
    """Gather requests that may share one pass, within the batching window."""
    first = self._carried or await self._queue.get()
    self._carried = None
    if first.abandoned:
      return []
    batch = [first]
    if not self._enabled or first.runs_alone:
      return batch
    deadline = asyncio.get_running_loop().time() + self._window_seconds
    lengths = [len(first.request.text)]
    while len(batch) < self._max_batch_size:
      remaining = deadline - asyncio.get_running_loop().time()
      if remaining <= 0:
        break
      try:
        queued = await asyncio.wait_for(self._queue.get(), remaining)
      except TimeoutError:
        break
      if queued.abandoned:
        continue
      if not self._may_join(queued, first, lengths):
        self._carried = queued
        break
      batch.append(queued)
      lengths.append(len(queued.request.text))
    return batch

  def _may_join(
    self,
    queued: _QueuedRequest,
    first: _QueuedRequest,
    lengths: list[int],
  ) -> bool:
    """Return whether a request can share the batch already being assembled.

    A batch runs until its longest item finishes, so pairing a very short
    request with a very long one wastes the difference. Requests that do not
    fit seed the next batch instead of being dropped or reordered.
    """
    if queued.runs_alone or queued.group_key != first.group_key:
      return False
    length = len(queued.request.text)
    shortest = min(min(lengths), length)
    longest = max(max(lengths), length)
    return longest <= shortest * LENGTH_RATIO

  async def _run_batch(self, batch: list[_QueuedRequest]) -> None:
    """Run one backend pass and resolve every request future."""
    active = [queued for queued in batch if not queued.abandoned]
    if not active:
      return
    first = active[0]
    try:
      async with self._generation_lock:
        generated = await self.backends.tts.generate_batch(
          [queued.request for queued in active],
          seed=first.seed,
          cancel_events=[queued.cancel_event for queued in active],
        )
      if len(generated) != len(active):
        raise RuntimeError(
          f"The backend returned {len(generated)} waveforms for {len(active)} requests."
        )
    except asyncio.CancelledError:
      for queued in active:
        if not queued.future.done():
          queued.future.cancel()
      raise
    except Exception as error:
      for queued in active:
        if not queued.future.done():
          queued.future.set_exception(error)
      return
    for queued, audio in zip(active, generated, strict=True):
      if not queued.future.done():
        queued.future.set_result(audio)
