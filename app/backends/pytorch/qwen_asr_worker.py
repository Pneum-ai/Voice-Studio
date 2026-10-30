"""Dependency-isolated JSON-line worker for Qwen3-ASR inference."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

RESPONSE_PREFIX = "VOICE_ASR:"
LANGUAGE_NAMES = {
  "ar": "Arabic",
  "cs": "Czech",
  "da": "Danish",
  "de": "German",
  "el": "Greek",
  "en": "English",
  "es": "Spanish",
  "fa": "Persian",
  "fi": "Finnish",
  "fil": "Filipino",
  "fr": "French",
  "hi": "Hindi",
  "hu": "Hungarian",
  "id": "Indonesian",
  "it": "Italian",
  "ja": "Japanese",
  "ko": "Korean",
  "mk": "Macedonian",
  "ms": "Malay",
  "nl": "Dutch",
  "pl": "Polish",
  "pt": "Portuguese",
  "ro": "Romanian",
  "ru": "Russian",
  "sv": "Swedish",
  "th": "Thai",
  "tr": "Turkish",
  "vi": "Vietnamese",
  "yue": "Cantonese",
  "zh": "Chinese",
}


def _write(payload: dict[str, object]) -> None:
  """Emit one response with a prefix resilient to dependency console output."""
  print(RESPONSE_PREFIX + json.dumps(payload, ensure_ascii=False), flush=True)


def _remove_worker_import_shadow() -> None:
  """Prevent the local qwen_asr adapter module from shadowing the installed package."""
  worker_directory = Path(__file__).resolve().parent
  sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != worker_directory]


def _load(model_path: str, device: str) -> Any:
  """Load a local Qwen3-ASR snapshot on the requested device."""
  _remove_worker_import_shadow()
  import torch
  from qwen_asr import Qwen3ASRModel  # pyright: ignore[reportAttributeAccessIssue]

  is_cuda = device.startswith("cuda") and torch.cuda.is_available()
  dtype = torch.bfloat16 if is_cuda else torch.float32
  device_map = device if is_cuda else "cpu"
  return Qwen3ASRModel.from_pretrained(
    model_path,
    dtype=dtype,
    device_map=device_map,
    max_inference_batch_size=1,
    max_new_tokens=4096,
  )


def main() -> None:
  """Read commands from standard input until the parent closes the worker."""
  model: Any | None = None
  for raw_line in sys.stdin:
    try:
      request = json.loads(raw_line)
      command = request.get("command")
      if command == "load":
        model = _load(str(request["model_path"]), str(request.get("device", "cpu")))
        _write({"ok": True})
      elif command == "transcribe":
        if model is None:
          raise RuntimeError("Qwen3-ASR model is not loaded.")
        language_code = request.get("language")
        language = LANGUAGE_NAMES.get(str(language_code), language_code)
        results = model.transcribe(
          audio=str(request["audio_path"]),
          context=str(request.get("context") or ""),
          language=language,
        )
        result = results[0]
        _write(
          {
            "ok": True,
            "text": str(result.text),
            "language": str(result.language) if result.language else None,
          }
        )
      else:
        raise ValueError(f"Unknown worker command: {command}")
    except Exception as error:
      _write({"ok": False, "error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
  main()
