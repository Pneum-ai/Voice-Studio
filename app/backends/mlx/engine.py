"""Bounded async delivery around one GPU-owned MLX speech scheduler."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from typing import Any

from app.backends.mlx.inference import Event, Inference, Request
from app.backends.mlx.worker import MLXWorker
from app.core.errors import InferenceTimeoutError, InferenceUnavailableError, OverloadedError
from app.utils.audio import AudioData

LOGGER = logging.getLogger(__name__)
MAX_REQUESTS = 32
MAX_BUFFERED_CHUNKS = 32
QUEUE_TIMEOUT_SECONDS = 30
REQUEST_TIMEOUT_SECONDS = 300


@dataclass(slots=True)
class Delivery:
  """Own event-loop-only output state for one live or saved request."""

  request: Request
  cancel: asyncio.Event
  created: float = field(default_factory=time.perf_counter)
  started: bool = False
  submitted: bool = False
  chunks: deque[AudioData] = field(default_factory=deque)
  ready: asyncio.Event = field(default_factory=asyncio.Event)
  finished: asyncio.Event = field(default_factory=asyncio.Event)
  error: Exception | None = None


class SpeechEngine:
  """Use the same queue, inference, cancellation, and PCM path for every speech request."""

  def __init__(self, model: Any, model_id: str, worker: MLXWorker) -> None:
    """Bind a loaded model without creating GPU state on the event loop."""
    self.model = model
    self.model_id = model_id
    self.worker = worker
    self.requests: dict[int, Delivery] = {}
    self.pending: list[Request] = []
    self.feedback: list[tuple[int, float, float]] = []
    self.next_identity = 0
    self.task: asyncio.Task[None] | None = None
    self.closing = False

  async def stream(
    self,
    *,
    text: str,
    reference: str | None,
    transcript: str | None,
    language: str,
    instructions: str | None,
    seed: int | None,
    cancel: asyncio.Event,
    speaker: str | None = None,
  ) -> AsyncGenerator[AudioData, None]:
    """Deliver bounded PCM and release the native row on cancellation or disconnect."""
    if self.closing:
      raise InferenceUnavailableError("the MLX model is unloading")
    if cancel.is_set():
      return
    if len(self.requests) >= MAX_REQUESTS:
      raise OverloadedError(1000)
    identity = self.next_identity
    self.next_identity += 1
    request = Request(identity, text, reference, transcript, language, instructions, seed, speaker)
    delivery = Delivery(request, cancel)
    self.requests[identity] = delivery
    self.pending.append(request)
    if self.task is None:
      self.task = asyncio.create_task(self._drive(), name="mlx-speech-engine")

    async def wake_on_cancel() -> None:
      """Wake a caller waiting for its first chunk when cancellation arrives."""
      await cancel.wait()
      delivery.ready.set()

    watcher = asyncio.create_task(wake_on_cancel())
    try:
      while True:
        limit = REQUEST_TIMEOUT_SECONDS if delivery.started else QUEUE_TIMEOUT_SECONDS
        try:
          await asyncio.wait_for(
            delivery.ready.wait(), max(0.001, limit - (time.perf_counter() - delivery.created))
          )
        except TimeoutError:
          self._fail(delivery, InferenceTimeoutError("The MLX speech deadline expired."))
        if delivery.error is not None:
          raise delivery.error
        if cancel.is_set():
          break
        if delivery.chunks:
          audio = delivery.chunks.popleft()
          if not delivery.finished.is_set():
            self.feedback.append(
              (identity, audio.samples.size / audio.sample_rate, time.perf_counter())
            )
          if not delivery.chunks and not delivery.finished.is_set():
            delivery.ready.clear()
          yield audio
        elif delivery.finished.is_set():
          break
        else:
          delivery.ready.clear()
    finally:
      if not delivery.finished.is_set():
        cancel.set()
        if not delivery.submitted:
          self.pending = [item for item in self.pending if item.identity != identity]
          self._finish(identity)
      watcher.cancel()
      await asyncio.gather(watcher, return_exceptions=True)
      await asyncio.shield(delivery.finished.wait())

  def _fail(self, delivery: Delivery, error: Exception) -> None:
    """Abort a failed consumer explicitly without blocking other listeners."""
    delivery.error = error
    delivery.chunks.clear()
    delivery.cancel.set()
    delivery.ready.set()

  def _finish(self, identity: int) -> None:
    """Mark native cleanup complete while allowing the consumer to drain buffered PCM."""
    delivery = self.requests.pop(identity, None)
    self.pending = [item for item in self.pending if item.identity != identity]
    if delivery is not None:
      delivery.finished.set()
      delivery.ready.set()

  def _deliver(self, event: Event) -> None:
    """Publish evaluated host audio; overflow fails only the slow consumer."""
    delivery = self.requests.get(event.identity)
    if delivery is None:
      return
    if event.started:
      delivery.started = True
    if event.error is not None:
      self._fail(delivery, event.error)
    if event.audio is not None and not delivery.cancel.is_set():
      if len(delivery.chunks) >= MAX_BUFFERED_CHUNKS:
        self._fail(delivery, InferenceTimeoutError("The audio consumer stopped accepting chunks."))
      else:
        delivery.chunks.append(event.audio)
        delivery.ready.set()
    if event.done:
      self._finish(event.identity)

  async def _drive(self) -> None:
    """Reserve GPU time for playing speech and advance inference one step at a time."""
    try:
      async with self.worker.reserve() as run:
        native = await run(Inference, self.model, self.model_id)
        submitted: set[int] = set()
        try:
          while self.requests:
            now = time.perf_counter()
            for delivery in self.requests.values():
              limit = REQUEST_TIMEOUT_SECONDS if delivery.started else QUEUE_TIMEOUT_SECONDS
              if now - delivery.created > limit and delivery.error is None:
                self._fail(delivery, InferenceTimeoutError("The MLX speech deadline expired."))
            if self.worker.has_waiters and not submitted:
              break
            incoming = []
            if not self.worker.has_waiters:
              incoming, self.pending = self.pending, []
              for item in incoming:
                self.requests[item.identity].submitted = True
                submitted.add(item.identity)
            feedback, self.feedback = self.feedback, []
            cancelled = {key for key, value in self.requests.items() if value.cancel.is_set()}
            events = await run(native.tick, incoming, cancelled, feedback)
            for event in events:
              self._deliver(event)
              if event.done:
                submitted.discard(event.identity)
            await asyncio.sleep(0)
        finally:
          await run(native.close)
    except asyncio.CancelledError:
      self.closing = True
      for identity, delivery in list(self.requests.items()):
        self._fail(delivery, InferenceUnavailableError("the MLX scheduler stopped"))
        self._finish(identity)
      raise
    except Exception as error:
      LOGGER.exception("MLX speech scheduler failed")
      for identity, delivery in list(self.requests.items()):
        self._fail(delivery, error)
        self._finish(identity)
    finally:
      self.task = None
      # New arrivals may race with the final GPU cleanup await.
      if self.requests and not self.closing:
        self.task = asyncio.create_task(self._drive(), name="mlx-speech-engine")
      elif not self.requests:
        self.pending.clear()
        self.feedback.clear()

  async def close(self) -> None:
    """Wake callers and await native cleanup before the loaded model is released."""
    self.closing = True
    for identity, delivery in list(self.requests.items()):
      self._fail(delivery, InferenceUnavailableError("the MLX model is unloading"))
      if not delivery.submitted:
        self._finish(identity)
    if self.task is not None:
      await asyncio.shield(self.task)
