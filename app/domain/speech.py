"""Speech model capabilities and explicit built-in speaker conditioning."""

from dataclasses import dataclass
from typing import Literal

CUSTOM_VOICE_MODEL = "qwen-tts-1.7B-customvoice"
QWEN_SPEAKERS = (
  "vivian",
  "serena",
  "uncle_fu",
  "dylan",
  "eric",
  "ryan",
  "aiden",
  "ono_anna",
  "sohee",
)


@dataclass(frozen=True, slots=True)
class SpeechCapabilities:
  """Describe whether a model clones references or accepts preset speakers and instructions."""

  voice_mode: Literal["clone", "builtin"] = "clone"
  speakers: tuple[str, ...] = ()
  instructions: bool = False


@dataclass(frozen=True, slots=True)
class BuiltinVoicePrompt:
  """Select a model-owned speaker without a reference waveform or profile."""

  speaker: str


@dataclass(frozen=True, slots=True)
class SpeechVoice:
  """Carry a display name and optional profile foreign key into saved generations."""

  id: str | None
  name: str
