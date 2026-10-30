"""PyTorch device selection shared by TTS and transcription."""

from __future__ import annotations

from typing import Any


def select_device(torch: Any) -> Any:
  """Return the best stable PyTorch device available on this host."""
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    return "cuda"
  try:
    import intel_extension_for_pytorch  # pyright: ignore[reportMissingImports]  # noqa: F401

    if hasattr(torch, "xpu") and torch.xpu.is_available():
      return "xpu"
  except ImportError:
    pass
  try:
    import torch_directml  # pyright: ignore[reportMissingImports]

    if torch_directml.device_count() > 0:
      return torch_directml.device(0)
  except ImportError:
    pass
  return "cpu"
