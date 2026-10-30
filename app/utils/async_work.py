"""Cancellation-friendly execution for blocking inference work."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from collections.abc import Callable
from functools import partial
from typing import Any


def _complete_future[Result](
  future: asyncio.Future[Result],
  *,
  result: Result | None = None,
  error: BaseException | None = None,
) -> None:
  """Complete a pending future without racing task cancellation."""
  if future.done():
    return
  if error is not None:
    future.set_exception(error)
  else:
    future.set_result(result)  # type: ignore[arg-type]


async def run_in_daemon_thread[Result](
  function: Callable[..., Result],
  *args: Any,
  thread_name: str = "voice-inference",
  **kwargs: Any,
) -> Result:
  """Run blocking model work without making process exit wait on its thread."""
  loop = asyncio.get_running_loop()
  future: asyncio.Future[Result] = loop.create_future()
  context = contextvars.copy_context()

  def run() -> None:
    """Execute the function and safely publish its outcome to the event loop."""
    try:
      result = context.run(function, *args, **kwargs)
    except BaseException as error:
      try:
        loop.call_soon_threadsafe(partial(_complete_future, future, error=error))
      except RuntimeError:
        return
    else:
      try:
        loop.call_soon_threadsafe(partial(_complete_future, future, result=result))
      except RuntimeError:
        return

  threading.Thread(
    target=run,
    name=thread_name,
    daemon=True,
  ).start()
  return await future
