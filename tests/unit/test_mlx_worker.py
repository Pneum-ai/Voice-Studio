"""Thread affinity and cancellation contracts for serialized MLX execution."""

from __future__ import annotations

import asyncio
import threading

import pytest

from app.backends.mlx.worker import MLXWorker


async def test_worker_keeps_thread_local_state_between_calls() -> None:
  """Preserve stream-like state across separately scheduled operations."""
  worker = MLXWorker()
  state = threading.local()

  def initialize() -> int:
    """Record state only visible to the owning thread."""
    state.value = 42
    return threading.get_ident()

  def read() -> tuple[int, int]:
    """Read state from a later operation on the same worker."""
    return threading.get_ident(), state.value

  owner = await worker.run(initialize)
  results = await asyncio.gather(*(worker.run(read) for _ in range(8)))
  assert owner != threading.get_ident()
  assert results == [(owner, 42)] * 8


async def test_worker_survives_an_operation_error() -> None:
  """Propagate a failed operation without discarding its owning thread."""
  worker = MLXWorker()
  owner = await worker.run(threading.get_ident)

  def fail() -> None:
    """Represent an inference error that should leave the worker usable."""
    raise ValueError("Invalid input")

  with pytest.raises(ValueError, match="Invalid input"):
    await worker.run(fail)
  assert await worker.run(threading.get_ident) == owner


async def test_cancellation_does_not_overlap_running_gpu_work() -> None:
  """Keep subsequent operations queued until cancelled native work returns."""
  worker = MLXWorker()
  started = threading.Event()
  release = threading.Event()
  order: list[str] = []

  def block() -> None:
    """Hold the worker until the caller releases this native-call substitute."""
    started.set()
    assert release.wait(timeout=5)
    order.append("finished")

  task = asyncio.create_task(worker.run(block))
  try:
    assert await asyncio.to_thread(started.wait, 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
      await task
    following = asyncio.create_task(worker.run(order.append, "next"))
    await asyncio.sleep(0)
    assert not following.done()
  finally:
    release.set()
  await following
  assert order == ["finished", "next"]
