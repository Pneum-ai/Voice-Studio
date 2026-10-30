"""PyTorch Whisper preprocessing and generation regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from transformers import WhisperFeatureExtractor

from app.backends.pytorch import stt
from app.utils.audio import AudioData


class _Processor:
  """Decode synthetic tokens without downloading tokenizer weights."""

  def batch_decode(
    self,
    _: torch.Tensor,
    *,
    skip_special_tokens: bool,
  ) -> list[str]:
    """Return one deterministic decoded transcript."""
    assert skip_special_tokens is True
    return ["attention-aware transcript"]


class _Model:
  """Capture arguments passed to Whisper generation."""

  def __init__(self) -> None:
    """Initialize empty generation metadata."""
    self.kwargs: dict[str, Any] = {}

  def generate(self, _: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Capture generation arguments and return one token row."""
    self.kwargs = kwargs
    return torch.ones((1, 2), dtype=torch.long)


@pytest.mark.parametrize("seconds", [1, 30, 31, 65])
def test_transcription_preserves_audio_beyond_first_window(
  monkeypatch: pytest.MonkeyPatch,
  seconds: int,
) -> None:
  """Use real Whisper features to detect dropped audio after the 30-second boundary."""
  extractor = WhisperFeatureExtractor()

  class Processor(_Processor):
    """Extract real audio features without downloading model weights or a tokenizer."""

    def __call__(self, samples: np.ndarray, **options: Any) -> Any:
      """Retain the real extracted features for checking full audio coverage."""
      self.inputs = extractor(samples, **options)
      return self.inputs

  backend = cast(Any, object.__new__(stt.PyTorchSTTBackend))
  processor = Processor()
  model = _Model()
  backend._torch = torch
  backend._device = "cpu"
  backend._model = model
  backend._processor = processor
  backend._model_dtype = torch.float32
  backend._generate_parameters = set()
  backend._generate_accepts_kwargs = True
  samples = np.zeros(seconds * 16_000, dtype=np.float32)
  # Put the only signal at the end, where the old default silently discarded it.
  samples[-8_000:] = np.sin(np.arange(8_000) * 0.1).astype(np.float32)
  monkeypatch.setattr(
    stt,
    "load_audio",
    lambda _path, sample_rate: AudioData(samples, sample_rate),
  )

  backend._transcribe_sync(Path("unused.wav"), "en", None)

  features = processor.inputs["input_features"]
  mask = processor.inputs["attention_mask"]
  assert torch.equal(model.kwargs["attention_mask"], mask)
  assert int(mask.sum()) == seconds * 100
  assert features.shape[-1] == max(seconds, 30) * 100
  assert features[..., seconds * 100 - 25].max() > features[..., 0].max()
  assert bool(model.kwargs.get("return_timestamps")) == (seconds > 30)
