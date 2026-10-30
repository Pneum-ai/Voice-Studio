"""Shared and exclusive access coordination for process-wide model memory."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class SharedExclusiveGuard:
  """Admit concurrent shared users while serializing exclusive transitions.

  Inference requests take shared access so they can overlap, while lifecycle
  work that replaces model memory takes exclusive access. Waiting exclusive
  holders block new shared users so a steady request stream cannot starve a
  pending load, unload, or delete.
  """

  def __init__(self) -> None:
    """Start with no shared users, no active holder, and no waiting holder."""
    self._condition = asyncio.Condition()
    self._shared_users = 0
    self._exclusive = False
    self._waiting_exclusive = 0

  @asynccontextmanager
  async def shared(self) -> AsyncIterator[None]:
    """Hold shared access, yielding first to any pending exclusive holder."""
    async with self._condition:
      await self._condition.wait_for(
        lambda: not self._exclusive and self._waiting_exclusive == 0,
      )
      self._shared_users += 1
    try:
      yield
    finally:
      async with self._condition:
        self._shared_users -= 1
        if self._shared_users == 0:
          self._condition.notify_all()

  @asynccontextmanager
  async def exclusive(self) -> AsyncIterator[None]:
    """Hold exclusive access once every shared user has finished."""
    async with self._condition:
      self._waiting_exclusive += 1
      try:
        await self._condition.wait_for(
          lambda: not self._exclusive and self._shared_users == 0,
        )
      except BaseException:
        self._waiting_exclusive -= 1
        self._condition.notify_all()
        raise
      self._waiting_exclusive -= 1
      self._exclusive = True
    try:
      yield
    finally:
      async with self._condition:
        self._exclusive = False
        self._condition.notify_all()
