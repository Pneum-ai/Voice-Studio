"""Model catalog and runtime state entities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from app.domain.speech import SpeechCapabilities


class ModelKind(StrEnum):
  """Distinguish speech generation and transcription models."""

  TTS = "tts"
  TRANSCRIPTION = "transcription"


class TranscriptionLanguageMode(StrEnum):
  """Describe how a recognition model accepts language selection."""

  AUTOMATIC = "automatic"
  OPTIONAL = "optional"
  REQUIRED = "required"


class DesiredModelState(StrEnum):
  """Describe a caller's requested local model state."""

  ABSENT = "absent"
  DOWNLOADED = "downloaded"
  LOADED = "loaded"


@dataclass(frozen=True, slots=True)
class TranscriptionCapabilities:
  """Describe model-specific recognition controls exposed to API clients."""

  languages: tuple[str, ...]
  language_mode: TranscriptionLanguageMode
  context: bool = False
  timestamps: tuple[str, ...] = ()
  streaming: bool = False
  translation: bool = False
  max_audio_seconds: int | None = None


@dataclass(frozen=True, slots=True)
class ModelDefinition:
  """Describe one selectable model independent of runtime state."""

  id: str
  display_name: str
  kind: ModelKind
  family: str
  provider: str
  transcription: TranscriptionCapabilities | None = None
  speech: SpeechCapabilities | None = None


@dataclass(frozen=True, slots=True)
class ModelStatus:
  """Describe download and memory state for one model."""

  id: str
  display_name: str
  kind: ModelKind
  family: str
  provider: str
  transcription: TranscriptionCapabilities | None
  downloaded: bool
  loaded: bool
  size_bytes: int | None
  backend_supported: bool
  native_streaming: bool
  active_job_id: str | None
  speech: SpeechCapabilities | None = None
