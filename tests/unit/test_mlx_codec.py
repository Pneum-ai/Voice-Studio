"""CPU regression tests for codec state isolation through joins and completion."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.backends.mlx.codec import CodecBank


class FakeCache:
  """Represent KV tensors so the test exercises both cache rows and convolution state."""

  def __init__(self) -> None:
    """Start with no previous token history."""
    self.state: tuple[np.ndarray, np.ndarray] | None = None


class FakeDecoder:
  """Use three independent mutable histories to detect incomplete snapshot handling."""

  def __init__(self) -> None:
    """Expose the same streaming field names as the pinned decoder."""
    self.pre_transformer = SimpleNamespace(make_cache=lambda: [FakeCache()])
    self.reset_streaming_state()

  def named_modules(self) -> list[tuple[str, Any]]:
    """Expose the causal fields to snapshot discovery."""
    return [("", self)]

  def reset_streaming_state(self) -> None:
    """Clear all request-dependent state while retaining model behavior."""
    self._buffer: np.ndarray | None = None
    self._overflow: np.ndarray | None = None
    self._transformer_cache: list[FakeCache] | None = None

  def streaming_step(self, codes: np.ndarray) -> np.ndarray:
    """Make wrong-row histories observable in the output samples."""
    values = codes[:, 0, :]
    if self._transformer_cache is None:
      self._transformer_cache = [FakeCache()]
    cache = self._transformer_cache[0]
    past = cache.state[0] if cache.state else np.zeros((values.shape[0], 1, 0, 1))
    waveform = values.cumsum(axis=1) + past.sum(axis=(1, 2, 3))[:, None]
    if self._buffer is not None:
      waveform += self._buffer[:, 0, 0][:, None]
    if self._overflow is not None:
      waveform += self._overflow[:, 0, 0][:, None]
    self._buffer = values[:, -1:, None]
    self._overflow = waveform[:, -1:, None] * 0.5
    history = np.concatenate((past, values[:, None, :, None]), axis=2)
    cache.state = (history, history)
    return waveform[:, None, :]


def install_arrays(monkeypatch: pytest.MonkeyPatch) -> None:
  """Use NumPy for scheduling tests so CI does not require Apple Silicon."""
  core = SimpleNamespace(
    concatenate=np.concatenate, stack=np.stack, zeros_like=np.zeros_like, eval=lambda *args: None
  )
  monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=core))
  monkeypatch.setitem(sys.modules, "mlx.core", core)
  monkeypatch.setitem(
    sys.modules, "mlx_audio.tts.continuous", SimpleNamespace(TTSBatchEvent=SimpleNamespace)
  )


def request(index: int, values: list[float]) -> SimpleNamespace:
  """Build sequential token frames for one independently identifiable voice stream."""
  return SimpleNamespace(sequence_id=index, generated_codes=[np.array([[x]]) for x in values])


def test_interleaved_decoder_state_matches_independent_streams(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Interleaving old, new, and completed rows must preserve every state component."""
  install_arrays(monkeypatch)
  decoder = FakeDecoder()
  # The actual MLX compile wrapper also exposes bound methods without module attributes.
  wrapper = SimpleNamespace(streaming_step=decoder.streaming_step)
  bank = CodecBank(wrapper)
  first = request(10, [1, 2])
  second = request(20, [10, 20])
  outputs: dict[int, list[np.ndarray]] = {10: [], 20: [], 30: []}

  def collect(events: list[Any]) -> None:
    """Retain audio by request ID, independent of codec grouping order."""
    for event in events:
      outputs[event.sequence_id].append(event.audio)

  collect(bank.decode([(first, False), (second, False)], 24000))
  first.generated_codes.extend(request(0, [3, 4]).generated_codes)
  third = request(30, [100, 200])
  collect(bank.decode([(first, True), (third, False)], 24000))
  bank.forget(20)
  third.generated_codes.extend(request(0, [300, 400]).generated_codes)
  collect(bank.decode([(third, True)], 24000))
  for identity, values in [(10, [1, 2, 3, 4]), (30, [100, 200, 300, 400])]:
    independent = FakeDecoder()
    expected = [
      independent.streaming_step(np.array(values[i : i + 2])[None, None, :])[0, 0] for i in (0, 2)
    ]
    np.testing.assert_array_equal(np.concatenate(outputs[identity]), np.concatenate(expected))
  assert bank.states == {}
  assert decoder._transformer_cache is None
