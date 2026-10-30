"""Model-family registry for one-loaded-at-a-time transcription adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from app.backends.protocols import (
  DownloadProgressCallback,
  STTBackend,
  TranscriptionRequest,
  TranscriptionResult,
)
from app.core.errors import ValidationError


class STTRegistry:
  """Dispatch lifecycle and inference work to model-family adapters."""

  def __init__(self, adapters: Sequence[STTBackend]) -> None:
    """Store ordered adapters and serialize cross-family model switches."""
    self._adapters = tuple(adapters)
    self._lock = asyncio.Lock()

  @property
  def loaded_model(self) -> str | None:
    """Return the single transcription model currently held in memory."""
    loaded = [adapter.loaded_model for adapter in self._adapters if adapter.loaded_model]
    if len(loaded) > 1:
      raise RuntimeError("More than one transcription adapter is loaded.")
    return loaded[0] if loaded else None

  def _adapter(self, model_id: str) -> STTBackend:
    """Resolve a model ID to its supporting adapter."""
    for adapter in self._adapters:
      if adapter.supports(model_id):
        return adapter
    raise ValidationError(
      f"No installed transcription adapter supports {model_id}.",
      "unsupported_model",
    )

  def supports(self, model_id: str) -> bool:
    """Return whether an installed adapter can run the model."""
    return any(adapter.supports(model_id) for adapter in self._adapters)

  def is_cached(self, model_id: str) -> bool:
    """Return whether the selected adapter has complete local weights."""
    return self._adapter(model_id).is_cached(model_id)

  def cache_size(self, model_id: str) -> int | None:
    """Return the selected adapter's cached model size."""
    return self._adapter(model_id).cache_size(model_id)

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Download model weights through the selected adapter with progress."""
    await self._adapter(model_id).download(model_id, progress)

  async def load(self, model_id: str) -> None:
    """Unload the active family and load the selected transcription model."""
    target = self._adapter(model_id)
    async with self._lock:
      for adapter in self._adapters:
        if adapter is not target and adapter.loaded_model is not None:
          await adapter.unload()
      await target.load(model_id)

  async def unload(self) -> None:
    """Release every adapter, tolerating the normal all-unloaded state."""
    async with self._lock:
      await asyncio.gather(*(adapter.unload() for adapter in self._adapters))

  async def delete(self, model_id: str) -> None:
    """Remove one model from its owning adapter."""
    async with self._lock:
      await self._adapter(model_id).delete(model_id)

  async def transcribe(
    self,
    request: TranscriptionRequest,
  ) -> TranscriptionResult:
    """Dispatch recognition to the adapter holding the active model."""
    loaded = self.loaded_model
    if loaded is None:
      raise RuntimeError("No transcription model is loaded.")
    return await self._adapter(loaded).transcribe(request)
