"""Backend-facing live-text adapter kept separate from the core Qwen wrapper."""

from __future__ import annotations

import asyncio
from typing import Any

from app.backends.protocols import BatchGenerationRequest, LiveTextStream
from app.core.errors import ValidationError


class NariLiveBackendMixin:
  """Expose Nari phrase prefetch through the generic TTS backend contract."""

  def _nari_runtime(self) -> Any | None:
    """Return the active accelerated runtime when available."""
    raise NotImplementedError

  async def open_live_text(
    self,
    *,
    prompt: object,
    language: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
  ) -> LiveTextStream:
    """Open an appendable Nari request for the active Base model."""
    nari = self._nari_runtime()
    if nari is None:
      raise ValidationError(
        "Live text input requires the Nari inference engine.",
        code="live_text_not_supported",
      )
    return await nari.open_live_text(
      BatchGenerationRequest(
        text="",
        prompt=prompt,
        language=language,
        instructions=instructions,
      ),
      seed=seed,
      cancel_event=cancel_event,
    )


__all__ = ["NariLiveBackendMixin"]
