"""Dependency-isolated JSON-line worker for NVIDIA NeMo ASR inference."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from typing import Any

RESPONSE_PREFIX = "VOICE_NEMO_ASR:"


def _write(payload: dict[str, object]) -> None:
  """Emit one response with a prefix resilient to dependency console output."""
  print(RESPONSE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _load(model_path: str, model_id: str, device: str) -> Any:
  """Restore a local NeMo checkpoint on the requested device."""
  from nemo.collections.asr.models import ASRModel  # pyright: ignore[reportMissingImports]

  snapshot = Path(model_path)
  checkpoints = sorted(snapshot.glob("*.nemo")) or sorted(snapshot.rglob("*.nemo"))
  if not checkpoints:
    raise RuntimeError(f"No .nemo checkpoint was found in {snapshot}.")
  model = ASRModel.restore_from(str(checkpoints[0]), map_location=device)
  model = model.to(device)
  model.eval()
  if model_id.startswith("nvidia-parakeet") and hasattr(model, "change_attention_model"):
    model.change_attention_model(
      self_attention_model="rel_pos_local_attn",
      att_context_size=[256, 256],
    )
  return model


def _transcription_input(model: Any, audio_path: str) -> object:
  """Avoid NeMo's temporary manifest lifecycle for one Windows audio file."""
  if sys.platform != "win32" or not Path(audio_path).is_file():
    return [audio_path]
  import librosa
  import numpy as np

  preprocessor = getattr(model, "preprocessor", None)
  sample_rate = int(getattr(preprocessor, "_sample_rate", 16_000))
  waveform, _ = librosa.load(
    audio_path,
    sr=sample_rate,
    mono=True,
    dtype=np.float32,
  )
  return np.asarray(waveform, dtype=np.float32)


def _transcribe(
  model: Any,
  model_id: str,
  audio_path: str,
  language: str | None,
  context: str | None,
) -> dict[str, object]:
  """Translate generic recognition controls into NeMo model arguments."""
  kwargs: dict[str, Any] = {}
  if model_id.startswith("nvidia-canary"):
    if not language:
      raise ValueError("Canary requires a source language.")
    kwargs.update(source_lang=language, target_lang=language)
    parameters = inspect.signature(model.transcribe).parameters
    accepts_keywords = any(
      parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if context:
      if "decodercontext" not in parameters and not accepts_keywords:
        raise ValueError("The installed NeMo runtime does not expose Canary context.")
      kwargs["decodercontext"] = context
  outputs = model.transcribe(_transcription_input(model, audio_path), **kwargs)
  output = outputs[0]
  text = str(getattr(output, "text", output)).strip()
  detected = getattr(output, "language", None) or language
  return {
    "ok": True,
    "text": text,
    "language": str(detected) if detected else None,
  }


def main() -> None:
  """Read commands from standard input until the parent closes the worker."""
  model: Any | None = None
  model_id: str | None = None
  for raw_line in sys.stdin:
    try:
      request = json.loads(raw_line)
      command = request.get("command")
      if command == "load":
        model_id = str(request["model_id"])
        model = _load(
          str(request["model_path"]),
          model_id,
          str(request.get("device", "cpu")),
        )
        _write({"ok": True})
      elif command == "transcribe":
        if model is None or model_id is None:
          raise RuntimeError("NVIDIA ASR model is not loaded.")
        language = request.get("language")
        context = request.get("context")
        _write(
          _transcribe(
            model,
            model_id,
            str(request["audio_path"]),
            str(language) if language else None,
            str(context) if context else None,
          )
        )
      else:
        raise ValueError(f"Unknown worker command: {command}")
    except Exception as error:
      _write({"ok": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
  main()
