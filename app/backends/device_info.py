"""Runtime hardware inspection and concise startup diagnostics."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DeviceInfo:
  """Describe the selected inference runtime and detected accelerator."""

  device_name: str | None = None
  runtime_version: str | None = None
  accelerator_memory_bytes: int | None = None
  accelerator_advisory: str | None = None


def _nvidia_smi_device() -> tuple[str, int, str] | None:
  """Return the first NVIDIA device visible to the host driver."""
  executable = shutil.which("nvidia-smi")
  if not executable:
    return None
  try:
    result = subprocess.run(
      [
        executable,
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
      ],
      capture_output=True,
      check=True,
      text=True,
      timeout=3,
    )
  except (OSError, subprocess.SubprocessError):
    return None
  lines = result.stdout.strip().splitlines()
  if not lines:
    return None
  first_line = lines[0]
  parts = [part.strip() for part in first_line.split(",")]
  if len(parts) != 3:
    return None
  try:
    memory_bytes = int(parts[1]) * 1024 * 1024
  except ValueError:
    memory_bytes = 0
  return parts[0], memory_bytes, parts[2]


def inspect_pytorch_device(torch: Any, device: Any) -> DeviceInfo:
  """Inspect the selected PyTorch device and any unusable NVIDIA hardware."""
  device_label = str(device)
  runtime_version = str(torch.__version__)
  if device_label.startswith("cuda") and torch.cuda.is_available():
    index = getattr(device, "index", None) or 0
    properties = torch.cuda.get_device_properties(index)
    return DeviceInfo(
      device_name=str(properties.name),
      runtime_version=runtime_version,
      accelerator_memory_bytes=int(properties.total_memory),
    )
  nvidia = _nvidia_smi_device()
  if nvidia and not torch.cuda.is_available():
    name, memory_bytes, driver = nvidia
    cuda_build = getattr(torch.version, "cuda", None)
    advisory = (
      f"{name} is visible through NVIDIA driver {driver}, but PyTorch "
      f"{runtime_version} has CUDA build {cuda_build or 'none'}. "
      "Run `uv sync` to restore Voice Open's default accelerated runtime."
    )
    return DeviceInfo(
      device_name=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
      runtime_version=runtime_version,
      accelerator_memory_bytes=memory_bytes,
      accelerator_advisory=advisory,
    )
  return DeviceInfo(
    device_name=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER"),
    runtime_version=runtime_version,
  )


def log_runtime_summary(
  *,
  backend: str,
  device: str,
  details: DeviceInfo,
  torch: Any | None = None,
) -> None:
  """Log platform, backend, accelerator, and actionable runtime details."""
  LOGGER.info(
    "Runtime: %s %s (%s) | Python %s",
    platform.system(),
    platform.release(),
    platform.machine(),
    sys.version.split()[0],
  )
  LOGGER.info(
    "Inference: backend=%s | device=%s | device_name=%s",
    backend,
    device,
    details.device_name or "unknown",
  )
  if torch is not None:
    LOGGER.info(
      "PyTorch: version=%s | CUDA build=%s | CUDA available=%s",
      details.runtime_version,
      getattr(torch.version, "cuda", None) or "none",
      torch.cuda.is_available(),
    )
  if details.accelerator_memory_bytes:
    LOGGER.info(
      "Detected accelerator memory: %.1f GiB",
      details.accelerator_memory_bytes / (1024**3),
    )
  if details.accelerator_advisory:
    LOGGER.warning("%s", details.accelerator_advisory)
