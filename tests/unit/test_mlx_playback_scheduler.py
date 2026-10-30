"""Check audio publication ordering without loading MLX or GPU weights."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from app.backends.mlx import session as serving


class FakeSession:
  """Supply token generation primitives while exercising the real scheduler."""

  def __init__(self, model: Any, options: Any) -> None:
    """Keep a finite queue and active rows with observable token histories."""
    self._active: list[Any] = []
    self._pending: list[Any] = []
    self.capacity = options.max_batch_size
    self.model = model

  @property
  def available_slots(self) -> int:
    """Expose capacity exactly as the upstream session contract does."""
    return self.capacity - len(self._active)

  def _advance_active(self) -> None:
    """Generate one token for each active request."""
    for state in self._active:
      state.generated_codes.append(0)

  def _take_pending_batch(self) -> list[Any]:
    """Remove the queued requests that fit into currently free slots."""
    batch = self._pending[: self.available_slots]
    del self._pending[: len(batch)]
    return batch

  def _admit_pending(self) -> None:
    """Model prefill producing its first audio token."""
    for state in self._take_pending_batch():
      state.generated_codes.append(0)
      self._active.append(state)


class FakeCodec:
  """Expose publication boundaries, independently of audio generation math."""

  def __init__(self, decoder: Any, *, mixed_ages: bool) -> None:
    """Start with no decoded frames."""
    self.states: dict[int, Any] = {}

  def decode(self, ready: list[tuple[Any, bool]], sample_rate: int) -> list[Any]:
    """Return the exact range of newly published token frames per listener."""
    events = []
    for state, done in ready:
      previous = self.states.get(state.sequence_id, SimpleNamespace(frames=0)).frames
      frames = len(state.generated_codes)
      self.states[state.sequence_id] = SimpleNamespace(frames=frames)
      events.append(
        SimpleNamespace(sequence_id=state.sequence_id, frames=frames - previous, done=done)
      )
    return events


def test_ready_audio_returns_before_prefill_and_admission_reserves_next_decode(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A newcomer cannot delay ready audio or postpone the decode promised by admission."""
  monkeypatch.setitem(
    sys.modules, "mlx_audio.tts.continuous", SimpleNamespace(TTSBatchOptions=SimpleNamespace)
  )
  monkeypatch.setitem(
    sys.modules,
    "mlx_audio.tts.models.qwen3_tts",
    SimpleNamespace(continuous_batching=SimpleNamespace(Qwen3TTSBatchSession=FakeSession)),
  )
  monkeypatch.setattr(serving, "CodecBank", FakeCodec)
  monkeypatch.setattr(serving.time, "perf_counter", lambda: 10.0)
  model = SimpleNamespace(
    _normalize_shared_batch_refs=lambda *args, **kwargs: ("voice.wav", "Reference."),
    speech_tokenizer=SimpleNamespace(decoder=None),
    sample_rate=24000,
  )
  session = serving.create_session(model, capacity=2)
  session._active = [SimpleNamespace(sequence_id=1, generated_codes=[0] * 3)]
  session._pending = [SimpleNamespace(sequence_id=2, generated_codes=[])]
  session.steps = 3
  session.playback_received(1, 1, now=10)
  events = session.step()
  assert [(event.sequence_id, event.frames) for event in events] == [(1, 4)]
  assert 2 not in session.admitted
  assert session.step() == []
  assert 2 in session.admitted
  # The regular four-step cadence is still two steps away, but admission reserved
  # a decode on this step. Both old and new rows must publish now.
  events = session.step()
  assert [(event.sequence_id, event.frames) for event in events] == [(1, 2), (2, 2)]
