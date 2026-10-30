"""Shared streaming onset detection for generated speech."""

from __future__ import annotations

import numpy as np

from app.utils.audio import AudioData


class LeadingSilenceTrimmer:
  """Drop only initial PCM before sustained speech while preserving short pre-roll."""

  def __init__(
    self,
    *,
    sample_rate: int,
    threshold_dbfs: float,
    max_ms: int,
    preroll_ms: int,
    min_output_ms: int = 0,
  ) -> None:
    """Configure the paper-compatible 20 ms window and 10 ms hop detector."""
    self._sample_rate = sample_rate
    self._threshold = 10 ** (threshold_dbfs / 20)
    self._window = sample_rate * 20 // 1000
    self._hop = sample_rate * 10 // 1000
    self._maximum = sample_rate * max_ms // 1000
    self._preroll = sample_rate * preroll_ms // 1000
    self._minimum_output = sample_rate * min_output_ms // 1000
    self._buffer = np.empty(0, dtype=np.float32)
    self._scan = 0
    self._active_windows = 0
    self._first_active: int | None = None
    self._onset: int | None = None
    self._opened = False

  @property
  def opened(self) -> bool:
    """Return whether initial onset detection has released the PCM stream."""
    return self._opened

  def process(self, audio: AudioData) -> AudioData | None:
    """Buffer initial chunks until speech is sustained, then pass through unchanged."""
    if self._opened:
      return audio
    if audio.sample_rate != self._sample_rate:
      raise ValueError("Leading-silence trimmer received an unexpected sample rate.")
    self._buffer = np.concatenate((self._buffer, audio.samples.astype(np.float32, copy=False)))
    if self._onset is not None:
      return self._open_if_buffered()
    while self._scan + self._window <= self._buffer.size:
      frame = self._buffer[self._scan : self._scan + self._window]
      centered = frame - float(np.mean(frame))
      rms = float(np.sqrt(np.mean(np.square(centered, dtype=np.float32))))
      if rms >= self._threshold:
        if self._active_windows == 0:
          self._first_active = self._scan
        self._active_windows += 1
        if self._active_windows >= 2:
          self._onset = max(0, (self._first_active or 0) - self._preroll)
          return self._open_if_buffered()
      else:
        self._active_windows = 0
        self._first_active = None
      self._scan += self._hop
    if self._onset is None and self._buffer.size >= self._maximum:
      return self._open(0)
    return None

  def finish(self) -> AudioData | None:
    """Preserve quiet or malformed output if the stream ends before detection."""
    if self._opened or not self._buffer.size:
      return None
    return self._open(self._onset or 0)

  def _open_if_buffered(self) -> AudioData | None:
    """Open after enough post-onset media exists for immediate playback."""
    assert self._onset is not None
    if self._buffer.size - self._onset < self._minimum_output:
      return None
    return self._open(self._onset)

  def _open(self, start: int) -> AudioData:
    """Open the one-way gate and return buffered PCM from the chosen onset."""
    samples = self._buffer[start:]
    self._buffer = np.empty(0, dtype=np.float32)
    self._opened = True
    return AudioData(samples, self._sample_rate)
