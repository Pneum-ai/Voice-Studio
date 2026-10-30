"""Keep MLX model state and GPU operations on one persistent daemon thread."""

from __future__ import annotations

import asyncio
import contextvars
import threading
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import Future
from contextlib import asynccontextmanager
from queue import Queue
from typing import Any


def _serve(queue: Queue[Callable[[], None] | None]) -> None:
  """Execute queued work until its owner is released, without retaining the owner."""
  while (work := queue.get()) is not None:
    work()
    del work


class MLXWorker:
  """Serialize model loading, inference, and release on the same GPU-owning thread."""

  def __init__(self) -> None:
    """Start a daemon so a stalled native call cannot prevent process shutdown."""
    self._queue: Queue[Callable[[], None] | None] = Queue()
    self._reservation = asyncio.Lock()
    self._waiting = 0
    threading.Thread(target=_serve, args=(self._queue,), name="voice-mlx", daemon=True).start()
    weakref.finalize(self, self._queue.put, None)

  async def run[Result](self, function: Callable[..., Result], *args: Any, **kwargs: Any) -> Result:
    """Wait behind active speech before submitting unrelated model or transcription work."""
    self._waiting += 1
    try:
      await self._reservation.acquire()
    finally:
      self._waiting -= 1
    try:
      return await self._run(function, *args, **kwargs)
    finally:
      self._reservation.release()

  @property
  def has_waiters(self) -> bool:
    """Let speech stop admitting newcomers while transcription waits for the GPU."""
    return self._waiting > 0

  @asynccontextmanager
  async def reserve(self) -> AsyncIterator[Callable[..., Awaitable[Any]]]:
    """Reserve the worker across speech steps so a long STT call cannot stall playback."""
    async with self._reservation:
      yield self._run

  async def _run[Result](
    self, function: Callable[..., Result], *args: Any, **kwargs: Any
  ) -> Result:
    """Run work with caller context; cancellation skips work that has not started."""
    future: Future[Result] = Future()
    context = contextvars.copy_context()

    def invoke() -> None:
      """Publish success or failure without killing the persistent worker."""
      if not future.set_running_or_notify_cancel():
        return
      try:
        future.set_result(context.run(function, *args, **kwargs))
      except BaseException as error:
        future.set_exception(error)

    self._queue.put(invoke)
    return await asyncio.wrap_future(future)
