"""Stateful batched MLX audio decoding with isolated per-request histories.

Derived from the MLX Audio state layout; see THIRD_PARTY_NOTICES.md. Different
request ages share decoder passes through explicit per-row positions and masks.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CodecState:
  """Retain one request's causal convolution buffers and attention-cache rows."""

  frames: int = 0
  buffers: list[Any] = field(default_factory=list)
  cache: list[Any] = field(default_factory=list)


class MixedAgeTransformer:
  """Use per-row rotary positions and padding masks for mixed-age decoder requests."""

  def __init__(self, transformer: Any) -> None:
    """Share original weights while changing only positional and masking logic."""
    self.transformer = transformer

  def make_cache(self) -> list[Any]:
    """Return ordinary per-request cache objects for snapshot storage."""
    return self.transformer.make_cache()

  def __call__(self, inputs: Any, mask: Any = None, cache: Any = None) -> Any:
    """Match upstream decoder math while allowing batched caches with distinct offsets."""
    import mlx.core as mx  # pyright: ignore[reportMissingImports]

    if cache is None or not hasattr(cache[0], "left_padding"):
      return self.transformer(inputs, mask=mask, cache=cache)
    model = self.transformer
    length = inputs.shape[1]
    x = model.input_proj(inputs)
    positions = cache[0].offset[:, None] + mx.arange(length)[None, :]
    rotary = model.rotary_emb(x, positions)
    keys = mx.arange(cache[0]._idx + length)
    causal = keys[None, :] <= (cache[0]._idx + mx.arange(length))[:, None]
    valid = keys[None, :] >= cache[0].left_padding[:, None]
    # Match the upstream additive mask, including its finite float32 sentinel.
    mask = mx.where(causal[None, :, :] & valid[:, None, :], 0, mx.finfo(mx.float32).min)
    mask = mask[:, None, :, :].astype(x.dtype)
    for layer, layer_cache in zip(model.layers, cache, strict=True):
      x = layer(x, rotary, mask, layer_cache)
    return model.output_proj(model.norm(x))


class CodecBank:
  """Move compatible request rows into the shared decoder on its owning GPU thread."""

  def __init__(self, decoder: Any, *, mixed_ages: bool = False) -> None:
    """Discover mutable streaming fields without duplicating decoder weights."""
    # mlx-audio wraps decoder.__call__ with mx.compile. Its bound streaming method
    # owns the real module; assigning state on the wrapper would affect the wrong object.
    decoder = decoder.streaming_step.__self__
    self.decoder = decoder
    self.mixed_ages = mixed_ages
    if mixed_ages and not isinstance(decoder.pre_transformer, MixedAgeTransformer):
      decoder.pre_transformer = MixedAgeTransformer(decoder.pre_transformer)
    self.make_cache = decoder.pre_transformer.make_cache
    self.fields = [
      (module, name)
      for _, module in decoder.named_modules()
      for name in ("_buffer", "_overflow")
      if hasattr(module, name)
    ]
    self.states: dict[int, CodecState] = {}

  def forget(self, sequence_id: int) -> None:
    """Release a completed or cancelled request's decoder history."""
    self.states.pop(sequence_id, None)

  def decode(self, ready: list[tuple[Any, bool]], sample_rate: int) -> list[Any]:
    """Decode new frames, preserving row identity through grouping and completion."""
    import mlx.core as mx  # pyright: ignore[reportMissingImports]
    from mlx_audio.tts.continuous import TTSBatchEvent  # pyright: ignore[reportMissingImports]

    groups: dict[tuple[int, int], list[tuple[Any, bool]]] = defaultdict(list)
    events = []
    for request, terminal in ready:
      state = self.states.setdefault(request.sequence_id, CodecState())
      count = len(request.generated_codes) - state.frames
      if count == 0:
        if terminal:
          events.append(TTSBatchEvent(sequence_id=request.sequence_id, done=True))
          self.forget(request.sequence_id)
        continue
      groups[(0 if self.mixed_ages else state.frames, count)].append((request, terminal))
    try:
      for (_, count), group in groups.items():
        states = [self.states[request.sequence_id] for request, _ in group]
        self._restore(states)
        codes = mx.concatenate(
          [
            mx.stack(request.generated_codes[self.states[request.sequence_id].frames :], axis=-1)
            for request, _ in group
          ]
        )
        waves = self.decoder.streaming_step(codes)
        mx.eval(waves)
        self._save(states, count)
        for index, (request, terminal) in enumerate(group):
          audio = waves[index, 0]
          events.append(
            TTSBatchEvent(
              sequence_id=request.sequence_id,
              audio=audio,
              sample_rate=sample_rate,
              samples=audio.size,
              done=terminal,
              is_streaming_chunk=True,
              is_final_chunk=terminal,
            )
          )
          if terminal:
            self.forget(request.sequence_id)
    finally:
      self.decoder.reset_streaming_state()
    return events

  def _restore(self, states: list[CodecState]) -> None:
    """Merge same-age cache rows and causal buffers before a decoder operation."""
    import mlx.core as mx  # pyright: ignore[reportMissingImports]

    self.decoder.reset_streaming_state()
    if not any(state.frames for state in states):
      return
    for index, (module, name) in enumerate(self.fields):
      buffers = [state.buffers[index] if state.buffers else None for state in states]
      template = next((buffer for buffer in buffers if buffer is not None), None)
      value = (
        None
        if template is None
        else mx.concatenate(
          [buffer if buffer is not None else mx.zeros_like(template) for buffer in buffers]
        )
      )
      setattr(module, name, value)
    cache = self.make_cache()
    rows = [state.cache if state.cache else self.make_cache() for state in states]
    for layer, merged in enumerate(cache):
      if self.mixed_ages:
        from mlx_audio.lm.models.cache import KVCache  # pyright: ignore[reportMissingImports]

        cache[layer] = KVCache.merge([row[layer] for row in rows])
      else:
        merged.state = tuple(
          mx.concatenate([row[layer].state[part] for row in rows]) for part in (0, 1)
        )
    self.decoder._transformer_cache = cache

  def _save(self, states: list[CodecState], frames: int) -> None:
    """Extract each decoder row before another cohort mutates the shared module."""
    import mlx.core as mx  # pyright: ignore[reportMissingImports]

    caches = self.decoder._transformer_cache
    padding = caches[0].left_padding.tolist() if hasattr(caches[0], "left_padding") else None
    for index, state in enumerate(states):
      state.frames += frames
      state.buffers = [
        None if (value := getattr(module, name)) is None else value[index : index + 1]
        for module, name in self.fields
      ]
      state.cache = self.make_cache()
      for target, source in zip(state.cache, self.decoder._transformer_cache, strict=True):
        if padding is not None:
          target.state = (
            source.keys[index : index + 1, :, padding[index] : source._idx],
            source.values[index : index + 1, :, padding[index] : source._idx],
          )
        else:
          target.state = tuple(array[index : index + 1] for array in source.state)
      mx.eval(
        [value for value in state.buffers if value is not None],
        [cache.state for cache in state.cache],
      )
