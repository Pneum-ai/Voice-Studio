"""Audio decoding, normalization, validation, and encoding helpers."""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from app.core.errors import ValidationError

TARGET_SAMPLE_RATE = 24_000


@dataclass(frozen=True, slots=True)
class AudioData:
  """Hold mono floating-point audio and its sample rate."""

  samples: np.ndarray
  sample_rate: int

  @property
  def duration_ms(self) -> int:
    """Return the rounded audio duration in milliseconds."""
    return round(len(self.samples) / self.sample_rate * 1000)


@dataclass(frozen=True, slots=True)
class PCM16AudioData:
  """Hold transport-ready mono PCM16 while decoding samples only on demand."""

  pcm_s16le: bytes
  sample_rate: int

  def __post_init__(self) -> None:
    """Reject malformed PCM and invalid sample rates at the backend boundary."""
    if len(self.pcm_s16le) % 2:
      raise ValueError("PCM16 audio must contain complete signed 16-bit samples.")
    if self.sample_rate <= 0:
      raise ValueError("PCM16 audio sample rate must be positive.")

  @property
  def samples(self) -> np.ndarray:
    """Decode normalized float32 samples for persistence or waveform processing."""
    return np.frombuffer(self.pcm_s16le, dtype="<i2").astype(np.float32) / 32767.0

  @property
  def duration_ms(self) -> int:
    """Return the rounded audio duration in milliseconds without decoding PCM."""
    return round((len(self.pcm_s16le) // 2) / self.sample_rate * 1000)


StreamAudioData = AudioData | PCM16AudioData


@dataclass(frozen=True, slots=True)
class AudioSelection:
  """Describe one validated time interval within an audio file."""

  start_seconds: float
  end_seconds: float

  @property
  def duration_seconds(self) -> float:
    """Return the selected interval duration."""
    return self.end_seconds - self.start_seconds


def load_audio(
  path: Path,
  *,
  sample_rate: int | None = TARGET_SAMPLE_RATE,
  offset_seconds: float = 0.0,
  duration_seconds: float | None = None,
) -> AudioData:
  """Decode mono float32 data; a null rate preserves the source for validation."""
  try:
    samples, resolved_rate = librosa.load(
      path,
      sr=sample_rate,
      mono=True,
      dtype=np.float32,
      offset=offset_seconds,
      duration=duration_seconds,
    )
  except Exception as error:
    raise ValidationError(f"Unable to decode audio: {error}", "invalid_audio") from error
  return AudioData(np.asarray(samples, dtype=np.float32), int(resolved_rate))


def audio_duration_seconds(path: Path) -> float:
  """Read source duration without decoding the complete waveform."""
  try:
    information = sf.info(path)
    duration = float(information.duration)
  except (RuntimeError, TypeError, ValueError):
    try:
      duration = float(librosa.get_duration(path=path))
    except Exception as error:
      raise ValidationError(
        f"Unable to inspect audio: {error}",
        "invalid_audio",
      ) from error
  if not np.isfinite(duration) or duration <= 0:
    raise ValidationError("Uploaded audio is empty.", "invalid_audio")
  return duration


def resolve_audio_selection(
  source_duration_seconds: float,
  *,
  start_seconds: float | None = None,
  end_seconds: float | None = None,
) -> AudioSelection:
  """Validate and clamp client trim coordinates to one source duration."""
  start = 0.0 if start_seconds is None else float(start_seconds)
  end = source_duration_seconds if end_seconds is None else float(end_seconds)
  if not all(np.isfinite(value) for value in (source_duration_seconds, start, end)):
    raise ValidationError("Audio trim coordinates must be finite.", "invalid_audio_trim")
  if source_duration_seconds <= 0 or start < 0 or end <= start:
    raise ValidationError("Audio trim range is empty or invalid.", "invalid_audio_trim")
  tolerance_seconds = 0.05
  if start >= source_duration_seconds or end > source_duration_seconds + tolerance_seconds:
    raise ValidationError(
      "Audio trim range exceeds the uploaded audio.",
      "invalid_audio_trim",
    )
  return AudioSelection(
    start_seconds=start,
    end_seconds=min(end, source_duration_seconds),
  )


def load_audio_selection(
  path: Path,
  *,
  start_seconds: float | None = None,
  end_seconds: float | None = None,
  sample_rate: int | None = TARGET_SAMPLE_RATE,
) -> tuple[AudioData, AudioSelection]:
  """Inspect, validate, and decode only one selected source interval."""
  source_duration = audio_duration_seconds(path)
  selection = resolve_audio_selection(
    source_duration,
    start_seconds=start_seconds,
    end_seconds=end_seconds,
  )
  audio = load_audio(
    path,
    sample_rate=sample_rate,
    offset_seconds=selection.start_seconds,
    duration_seconds=selection.duration_seconds,
  )
  return audio, selection


def normalize_audio(
  audio: AudioData,
  *,
  target_db: float = -20.0,
  peak_limit: float = 0.85,
) -> AudioData:
  """Apply one linear gain, prioritizing peak headroom over the target RMS loudness."""
  samples = np.asarray(audio.samples, dtype=np.float32)
  rms = float(np.sqrt(np.mean(samples**2))) if samples.size else 0.0
  if rms > 0:
    target_rms = 10 ** (target_db / 20)
    peak = float(np.max(np.abs(samples)))
    samples = samples * min(target_rms / rms, peak_limit / peak)
  samples = samples.astype(np.float32)
  return AudioData(samples, audio.sample_rate)


def validate_reference_audio(
  audio: AudioData,
  *,
  min_duration_seconds: float = 2.0,
  max_duration_seconds: float = 30.0,
  min_rms: float = 0.01,
) -> None:
  """Validate original-rate audio, before resampling can create harmless peak overshoot."""
  if not np.all(np.isfinite(audio.samples)):
    raise ValidationError("Reference audio contains invalid samples.", "invalid_audio")
  duration_seconds = len(audio.samples) / audio.sample_rate
  if duration_seconds < min_duration_seconds:
    raise ValidationError(
      f"Reference audio must be at least {min_duration_seconds:g} seconds.",
      "audio_too_short",
    )
  if duration_seconds > max_duration_seconds:
    raise ValidationError(
      f"Reference audio must be no longer than {max_duration_seconds:g} seconds.",
      "audio_too_long",
    )
  rms = float(np.sqrt(np.mean(audio.samples**2))) if audio.samples.size else 0.0
  if rms < min_rms:
    raise ValidationError(
      "Reference audio is silent or too quiet.",
      "audio_too_quiet",
    )
  peak = float(np.max(np.abs(audio.samples))) if audio.samples.size else 0.0
  # PCM saturation may never exceed 1.0. Repeated same-sign rail samples indicate
  # flattened peaks; isolated full-scale samples are not enough to reject a clip.
  positive = audio.samples >= 1.0 - 1 / 32768
  negative = audio.samples <= -1.0 + 1 / 32768
  rails = positive | negative
  flat = any(np.any(side[:-2] & side[1:-1] & side[2:]) for side in (positive, negative))
  if peak > 1.0 or (flat and float(np.mean(rails)) >= 0.001):
    raise ValidationError(
      "Reference audio has distorted peaks. Lower your microphone input level or move "
      "farther away, then record again. You can also trim out a loud section. "
      "Lowering playback volume cannot repair an already clipped recording.",
      "audio_clipping",
    )


def prepare_reference_audio(audio: AudioData) -> AudioData:
  """Validate source peaks, convert to 24 kHz, and retain safe headroom without clipping."""
  validate_reference_audio(audio)
  samples = librosa.resample(audio.samples, orig_sr=audio.sample_rate, target_sr=TARGET_SAMPLE_RATE)
  return normalize_audio(AudioData(samples, TARGET_SAMPLE_RATE))


def write_wav(path: Path, audio: AudioData) -> None:
  """Write floating-point audio to a standard PCM WAV file."""
  path.parent.mkdir(parents=True, exist_ok=True)
  sf.write(path, audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")


def wav_bytes(audio: AudioData) -> bytes:
  """Encode audio as in-memory PCM WAV bytes."""
  buffer = io.BytesIO()
  sf.write(buffer, audio.samples, audio.sample_rate, format="WAV", subtype="PCM_16")
  return buffer.getvalue()


def encoded_audio_bytes(audio: AudioData, response_format: str) -> bytes:
  """Encode a complete waveform in one supported OpenAI speech format.

  Raw ``pcm`` remains fixed at mono 24 kHz signed little-endian PCM16, matching
  the OpenAI speech API transport contract used by real-time clients.
  """
  if response_format == "pcm":
    return pcm_s16le_bytes(audio)
  formats = {
    "flac": ("FLAC", None),
    "mp3": ("MP3", None),
    "wav": ("WAV", "PCM_16"),
  }
  if response_format not in formats:
    raise ValueError(f"Unsupported audio response format: {response_format}")
  format_name, subtype = formats[response_format]
  buffer = io.BytesIO()
  sf.write(
    buffer,
    audio.samples,
    audio.sample_rate,
    format=format_name,
    subtype=subtype,
  )
  return buffer.getvalue()


def pcm_s16le_bytes(audio: StreamAudioData) -> bytes:
  """Encode normalized floating-point audio as raw signed 16-bit PCM."""
  if isinstance(audio, PCM16AudioData):
    return audio.pcm_s16le
  clipped = np.clip(audio.samples, -1.0, 1.0)
  pcm = np.round(clipped * 32767.0).astype("<i2")
  return pcm.tobytes()
