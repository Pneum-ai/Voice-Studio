"""Transport-ready PCM audio behavior."""

from __future__ import annotations

import numpy as np
import pytest

from app.utils.audio import PCM16AudioData, pcm_s16le_bytes


def test_pcm16_audio_reuses_transport_bytes_and_decodes_lazily() -> None:
  """Preserve Nari bytes exactly while retaining waveform compatibility."""
  encoded = np.array([-32767, 0, 16384, 32767], dtype="<i2").tobytes()
  audio = PCM16AudioData(encoded, 24_000)

  assert pcm_s16le_bytes(audio) is encoded
  assert audio.duration_ms == 0
  assert audio.samples.dtype == np.float32
  assert audio.samples.tolist() == pytest.approx([-1.0, 0.0, 16384 / 32767, 1.0])


def test_pcm16_audio_rejects_incomplete_samples() -> None:
  """Fail at the backend boundary instead of sending malformed WebSocket PCM."""
  with pytest.raises(ValueError, match="complete signed 16-bit samples"):
    PCM16AudioData(b"\x00", 24_000)
