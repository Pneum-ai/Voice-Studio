"""Bounded GPU cache for invariant Qwen Base voice-clone prompt embeddings."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class BasePromptCacheKey:
    """Identify prompt tensors whose contents remain valid while the model is loaded."""

    language: str
    reference_text: str
    prompt_key: str


@dataclass(frozen=True, slots=True)
class BasePromptEmbedding:
    """Hold the invariant portions of one Base ICL embedding override."""

    initial: torch.Tensor
    reference_semantic: torch.Tensor
    reference_audio: torch.Tensor
    semantic_terminal: torch.Tensor


@dataclass(frozen=True, slots=True)
class BasePromptCacheStats:
    """Report cache effectiveness without exposing cached prompt contents."""

    hits: int
    misses: int
    entries: int


class BasePromptEmbeddingCache:
    """Retain a small LRU of reusable prompt embeddings on the model device."""

    def __init__(self, *, capacity: int = 8) -> None:
        """Create a cache with a strict entry limit."""
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Base prompt cache capacity must be a positive integer")
        self._capacity = capacity
        self._items: OrderedDict[BasePromptCacheKey, BasePromptEmbedding] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: BasePromptCacheKey) -> BasePromptEmbedding | None:
        """Return and promote a cached embedding, recording hit statistics."""
        value = self._items.get(key)
        if value is None:
            self._misses += 1
            return None
        self._items.move_to_end(key)
        self._hits += 1
        return value

    def put(self, key: BasePromptCacheKey, value: BasePromptEmbedding) -> None:
        """Insert an embedding and evict the least recently used prompt."""
        self._items[key] = value
        self._items.move_to_end(key)
        while len(self._items) > self._capacity:
            self._items.popitem(last=False)

    def clear(self) -> None:
        """Invalidate all device tensors after model-weight or adapter changes."""
        self._items.clear()

    def stats(self) -> BasePromptCacheStats:
        """Return cumulative hit counters and the current resident entry count."""
        return BasePromptCacheStats(
            hits=self._hits,
            misses=self._misses,
            entries=len(self._items),
        )


def tensor_cache_identity(value: torch.Tensor) -> tuple[int, int, tuple[int, ...]]:
    """Return a cheap identity that changes after in-place tensor mutation."""
    try:
        version = value._version
    except RuntimeError:
        # PyTorch inference tensors reject version-counter access and cannot be
        # mutated outside inference mode. Adapter swaps clear the entire cache.
        version = 0
    return value.data_ptr(), version, tuple(value.shape)


__all__ = [
    "BasePromptCacheKey",
    "BasePromptCacheStats",
    "BasePromptEmbedding",
    "BasePromptEmbeddingCache",
    "tensor_cache_identity",
]
