"""Reference quality checks must distinguish source damage from conversion overshoot."""

from __future__ import annotations

import numpy as np
import pytest

from app.core.errors import ValidationError
from app.utils.audio import AudioData, normalize_audio, validate_reference_audio


def test_normalization_preserves_waveform_ratios_instead_of_flattening_peaks() -> None:
  """A high-crest recording receives one gain even when the RMS target is unreachable."""
  samples = np.full(1000, 0.01, dtype=np.float32)
  samples[0] = 1.0
  result = normalize_audio(AudioData(samples, 24000))
  np.testing.assert_allclose(result.samples, samples * 0.85)
  assert result.sample_rate == 24000


def test_isolated_full_scale_peak_is_not_sustained_saturation() -> None:
  """One loud sample cannot establish that a whole reference has flattened peaks."""
  samples = np.full(48000, 0.1, dtype=np.float32)
  samples[200] = 1.0
  validate_reference_audio(AudioData(samples, 24000))


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_invalid_float_samples_are_rejected(value: float) -> None:
  """Non-finite audio must not bypass loudness validation or reach persistence."""
  samples = np.full(48000, 0.1, dtype=np.float32)
  samples[200] = value
  with pytest.raises(ValidationError, match="invalid samples"):
    validate_reference_audio(AudioData(samples, 24000))
