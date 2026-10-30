"""Load the Qwen 12 Hz runtime without probing for the unused SoX CLI."""

from __future__ import annotations

import importlib
import shutil
import sys
from types import ModuleType
from typing import Any


def _skip_unused_sox_probe() -> None:
  """Stub the Qwen 25 Hz-only SoX binding when its executable is absent."""
  if shutil.which("sox") is None and "sox" not in sys.modules:
    sys.modules["sox"] = ModuleType("sox")


_skip_unused_sox_probe()
_qwen: Any = importlib.import_module("qwen_tts")
_modeling: Any = importlib.import_module("qwen_tts.core.models.modeling_qwen3_tts")

Qwen3TTSModel = _qwen.Qwen3TTSModel
Qwen3TTSTokenizer = _qwen.Qwen3TTSTokenizer
VoiceClonePromptItem = _qwen.VoiceClonePromptItem
mel_spectrogram = _modeling.mel_spectrogram

__all__ = [
  "Qwen3TTSModel",
  "Qwen3TTSTokenizer",
  "VoiceClonePromptItem",
  "mel_spectrogram",
]
