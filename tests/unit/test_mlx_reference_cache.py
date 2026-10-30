"""Verify session-local speaker caching preserves voice and prompt routing."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.backends.mlx.prompts import ReferenceModel


class ReferenceFixture:
  """Model deterministic reference extraction and per-text prompt construction."""

  def __init__(self) -> None:
    """Retain distinct reference objects and count expensive speaker extractions."""
    self.voices = {"first.wav": np.array([1.0, 3.0]), "second.wav": np.array([5.0, 7.0])}
    self.extractions = 0

  def _normalize_shared_batch_refs(
    self, count: int, *, ref_audio: str, ref_text: str
  ) -> tuple[np.ndarray, str]:
    """Resolve each synthetic reference to its stable waveform object."""
    return self.voices[ref_audio], ref_text

  def extract_speaker_embedding(self, audio: np.ndarray) -> np.ndarray:
    """Represent costly deterministic voice analysis with a distinguishable result."""
    self.extractions += 1
    return np.array([audio.mean(), audio.max()])

  def _prepare_icl_generation_inputs(
    self, text: str, *, ref_audio: np.ndarray, ref_text: str
  ) -> np.ndarray:
    """Combine variable text and voice features through the normal extraction hook."""
    return np.concatenate((self.extract_speaker_embedding(ref_audio), [len(text), len(ref_text)]))

  def _prepare_batch_inputs(self, texts: list[str], **kwargs: Any) -> np.ndarray:
    """Exercise method routing through batch and ICL preparation, like the real model."""
    return np.stack([self._prepare_icl_generation_inputs(text, **kwargs) for text in texts])


def test_speaker_cache_preserves_changed_text_and_isolates_sessions() -> None:
  """Repeated requests reuse one extraction; other voices and sessions remain independent."""
  model = ReferenceFixture()
  original_extract = model.extract_speaker_embedding
  first = ReferenceModel(model, "first.wav", "First reference.", cache_speaker=True)
  second = ReferenceModel(model, "second.wav", "Second reference.", cache_speaker=True)
  np.testing.assert_array_equal(
    first._prepare_batch_inputs(["Hi", "Longer"]), [[2, 3, 2, 16], [2, 3, 6, 16]]
  )
  np.testing.assert_array_equal(first._prepare_batch_inputs(["New text"]), [[2, 3, 8, 16]])
  assert model.extractions == 1
  np.testing.assert_array_equal(second._prepare_batch_inputs(["Hi"]), [[6, 7, 2, 17]])
  assert model.extractions == 2
  np.testing.assert_array_equal(first._prepare_batch_inputs(["Hi"]), [[2, 3, 2, 16]])
  assert model.extractions == 2
  # A session cache cannot leak through replacement of the shared model's methods.
  assert model.extract_speaker_embedding == original_extract
  np.testing.assert_array_equal(first.extract_speaker_embedding(model.voices["second.wav"]), [6, 7])
  assert model.extractions == 3
  fresh = ReferenceModel(model, "first.wav", "First reference.", cache_speaker=True)
  np.testing.assert_array_equal(fresh._prepare_batch_inputs(["Hi"]), [[2, 3, 2, 16]])
  assert model.extractions == 4
