"""Continuous mixed-voice MLX decoding with persistent per-request state.

Uses the pinned mlx-audio primitives; upstream MIT attribution is retained in
THIRD_PARTY_NOTICES.md. All methods run on the GPU-owning worker.
"""

from __future__ import annotations

import time
from typing import Any

from app.backends.mlx.codec import CodecBank
from app.backends.mlx.playback import PlaybackPolicy
from app.backends.mlx.prompts import PromptModel
from app.core.errors import InferenceTimeoutError


def create_session(model: Any, *, capacity: int = 8, cache_speaker: bool = False) -> Any:
  """Create the single serving session; seeded requests use capacity one."""
  from mlx_audio.tts.continuous import TTSBatchOptions  # pyright: ignore[reportMissingImports]
  from mlx_audio.tts.models.qwen3_tts import continuous_batching as cb  # type: ignore

  chunk_frames = 4

  class CloneSession(cb.Qwen3TTSBatchSession):
    """Advance all active Talkers together, then decode only ready request cohorts."""

    def __init__(self) -> None:
      """Initialize upstream continuous state and request-specific codec storage."""
      super().__init__(
        PromptModel(model, cache_speaker=cache_speaker),
        TTSBatchOptions(
          max_batch_size=capacity, max_tokens=4096, repetition_penalty=1.5, lang_code="auto"
        ),
      )
      self.codec = CodecBank(model.speech_tokenizer.decoder, mixed_ages=True)
      self.cached_ids: list[int] = []
      self.batch_cache: list[Any] = []
      self.steps = 0
      self.decode_after_admission = False
      self.playback = PlaybackPolicy()
      self.terminal: list[Any] = []
      self.terminal_errors: dict[int, Exception] = {}
      self.admitted: list[int] = []

    def _take_pending_batch(self) -> list[Any]:
      """Bind per-row conditioning before upstream constructs the padded prompt batch."""
      batch = super()._take_pending_batch()
      self.admitted.extend(item.sequence_id for item in batch)
      self.model.preparing.extend(batch)
      return batch

    def _decode_state(self, state: Any) -> Any:
      """Defer terminal audio to the same codec grouping pass as active audio."""
      if len(state.generated_codes) >= self.options.max_tokens:
        self.terminal_errors[state.sequence_id] = InferenceTimeoutError(
          "Speech exceeded the model token limit; output was not saved as complete."
        )
      self.terminal.append(state)
      return self._empty_event(state.sequence_id)

    def _clear_cache_if_idle(self) -> None:
      """Keep GPU allocations warm across requests; cleanup belongs to the engine."""
      return None

    def playback_received(self, sequence_id: int, duration: float, now: float) -> None:
      """Receive caller feedback only for audio released after trimming / buffering."""
      if any(state.sequence_id == sequence_id for state in self._active):
        self.playback.received(sequence_id, duration, now)

    def step(self) -> list[Any]:
      """Publish ready audio before permitting new prefill to consume GPU time."""
      if self._active:
        started = time.perf_counter()
        self._advance_active()
        self.playback.observe("token", time.perf_counter() - started)
      self.steps += 1
      events = self._decode_ready(force=self.decode_after_admission)
      self.decode_after_admission = False
      if events:
        # Return immediately so the consumer can play it and report real headroom.
        return events
      rows = min(self.available_slots, len(self._pending))
      if rows and self.playback.can_admit(
        (state.sequence_id for state in self._active),
        rows,
        time.perf_counter(),
        1,
      ):
        started = time.perf_counter()
        self._admit_pending()
        self.playback.observe("prefill", time.perf_counter() - started, rows)
        # Admission reserves one token step and a decode, not the full cadence.
        # Publish on the next step even when no row otherwise reaches its deadline.
        self.decode_after_admission = True
        return self._decode_ready()
      return []

    def _decode_ready(self, *, force: bool = False) -> list[Any]:
      """Flush final audio and batch ready or playback-urgent request rows."""
      ready = [(state, True) for state in self.terminal]
      self.terminal.clear()
      now = time.perf_counter()
      for state in self._active:
        previous = self.codec.states.get(state.sequence_id)
        decoded = previous.frames if previous is not None else 0
        pending = len(state.generated_codes) - decoded
        scheduled = pending > 0 and self.steps % chunk_frames == 0
        urgent = self.playback.decode_due(
          state.sequence_id, pending, decoded, now, chunk_frames - self.steps % chunk_frames
        )
        if scheduled or urgent or (force and pending > 0):
          ready.append((state, False))
      if not ready:
        return []
      started = time.perf_counter()
      events = self.codec.decode(ready, model.sample_rate)
      self.playback.observe("codec", time.perf_counter() - started)
      for event in events:
        if event.done:
          self.playback.forget(event.sequence_id)
          event.error = self.terminal_errors.pop(event.sequence_id, None)
      return events

    def _merge_state_caches(self, states: list[Any]) -> list[Any]:
      """Reuse cache allocations; filter or extend rows only when membership changes."""
      ids = [state.sequence_id for state in states]
      keep = [index for index, identity in enumerate(self.cached_ids) if identity in ids]
      if not keep:
        self.batch_cache = super()._merge_state_caches(states)
        self.cached_ids = ids
        return self.batch_cache
      retained = [self.cached_ids[index] for index in keep]
      if len(keep) != len(self.cached_ids):
        for cache in self.batch_cache:
          cache.filter(keep)
      added = [state for state in states if state.sequence_id not in retained]
      if added:
        extra = super()._merge_state_caches(added)
        for cache, extension in zip(self.batch_cache, extra, strict=True):
          cache.extend(extension)
      assert retained + [state.sequence_id for state in added] == ids
      self.cached_ids = ids
      return self.batch_cache

    def _extract_batch_cache_rows(self, cache: list[Any], batch_size: int) -> list[list[Any]]:
      """Avoid extracting copies after every token when the persistent cache owns state."""
      if cache is self.batch_cache:
        return [state.cache for state in self._active]
      return super()._extract_batch_cache_rows(cache, batch_size)

    def _attention_mask_for_states(self, states: list[Any]) -> Any:
      """Mask cache padding using current slots, not stale per-request prefill snapshots."""
      import mlx.core as mx  # pyright: ignore[reportMissingImports]

      cache = self.batch_cache[0]
      return mx.arange(cache._idx + 1)[None, :] >= cache.left_padding[:, None]

    def cancel(self, sequence_id: int) -> None:
      """Remove queued or active work and all retained codec state."""
      super().cancel(sequence_id)
      self.codec.forget(sequence_id)
      self.playback.forget(sequence_id)
      self.terminal_errors.pop(sequence_id, None)
      self.terminal = [state for state in self.terminal if state.sequence_id != sequence_id]
      if not self._active:
        self.batch_cache = []
        self.cached_ids = []

    def close(self) -> None:
      """Release pending, active, and decoder state on the owning GPU thread."""
      self._pending.clear()
      self._active.clear()
      self.terminal.clear()
      self.codec.states.clear()
      self.playback.ends.clear()
      self.model.clear()
      self.terminal_errors.clear()
      self.batch_cache.clear()
      self.cached_ids.clear()
      self.codec.decoder.reset_streaming_state()

  return CloneSession()
