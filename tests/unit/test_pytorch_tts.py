"""PyTorch Qwen generation limits and cancellation behavior."""

from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

from app.backends.pytorch.qwen_generation import QWEN_MAX_NEW_TOKENS
from app.backends.pytorch.tts import PyTorchTTSBackend, qwen_generation_token_limit


class _QwenWrapper:
  """Capture wrapper arguments without loading model weights."""

  def __init__(self) -> None:
    """Start without captured generation arguments."""
    self.kwargs: dict[str, Any] = {}

  def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
    """Capture generation arguments and return deterministic audio."""
    self.kwargs = kwargs
    return [np.zeros(240, dtype=np.float32)], 24_000


def test_qwen_generation_limit_scales_but_never_uses_upstream_runaway_default() -> None:
  """Leave normal speech headroom while bounding a missed-EOS generation."""
  assert qwen_generation_token_limit("Hello.") == 96
  assert qwen_generation_token_limit("one two three four five six seven eight nine ten") == 128
  assert qwen_generation_token_limit("long " * 1_000) == QWEN_MAX_NEW_TOKENS
  assert QWEN_MAX_NEW_TOKENS < 2_048


@pytest.mark.asyncio
async def test_stream_cancels_qwen_inside_token_generation() -> None:
  """Propagate a WebSocket interruption into the blocking Transformers loop."""

  class BlockingTalker:
    """Run until the backend-injected stopping criterion observes cancellation."""

    def __init__(self) -> None:
      """Create synchronization markers for the blocking generation loop."""
      self.started = threading.Event()
      self.stopped = threading.Event()

    def generate(self, **kwargs: Any) -> object:
      """Block until the injected stopping criteria requests cancellation."""
      criteria = kwargs["stopping_criteria"]
      input_ids = torch.ones((1, 1), dtype=torch.long)
      self.started.set()
      while not bool(criteria(input_ids, None).all()):
        time.sleep(0.005)
      self.stopped.set()
      return object()

  class CancelableWrapper(_QwenWrapper):
    """Exercise the private talker call used by the installed Qwen wrapper."""

    def __init__(self) -> None:
      """Create a wrapper containing the blocking talker."""
      super().__init__()
      self.talker = BlockingTalker()
      self.model = SimpleNamespace(talker=self.talker)

    def generate_voice_clone(self, **kwargs: Any) -> tuple[list[np.ndarray], int]:
      """Capture arguments and enter the cancellable talker loop."""
      self.kwargs = kwargs
      self.model.talker.generate()
      return [np.zeros(240, dtype=np.float32)], 24_000

  backend = cast(Any, object.__new__(PyTorchTTSBackend))
  wrapper = CancelableWrapper()
  backend._torch = torch
  backend._model = wrapper
  cancel_event = asyncio.Event()

  async def collect() -> list[object]:
    """Collect every emitted stream chunk."""
    return [
      chunk
      async for chunk in backend.stream(
        text="Please stop when interrupted.",
        prompt={"voice": "prompt"},
        language="en",
        seed=None,
        instructions=None,
        cancel_event=cancel_event,
      )
    ]

  task = asyncio.create_task(collect())
  assert await asyncio.wait_for(asyncio.to_thread(wrapper.talker.started.wait, 10), timeout=11)
  cancel_event.set()

  chunks = await asyncio.wait_for(task, timeout=1)
  assert chunks == []
  assert wrapper.talker.stopped.is_set()
  assert wrapper.kwargs["max_new_tokens"] == qwen_generation_token_limit(
    "Please stop when interrupted."
  )
