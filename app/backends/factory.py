"""Platform detection and backend bundle construction."""

from __future__ import annotations

import importlib.util
import platform
from dataclasses import dataclass
from pathlib import Path

from app.backends.device_info import (
  DeviceInfo,
  inspect_pytorch_device,
  log_runtime_summary,
)
from app.backends.protocols import STTBackend, TTSBackend
from app.backends.pytorch.nari_streaming import NariRuntimeConfig
from app.backends.stt_registry import STTRegistry


@dataclass(frozen=True, slots=True)
class BackendBundle:
  """Hold the selected TTS and STT implementations and device labels."""

  name: str
  device: str
  tts: TTSBackend
  stt: STTBackend
  device_info: DeviceInfo = DeviceInfo()


def is_apple_silicon() -> bool:
  """Return whether the host is an Apple Silicon Mac."""
  return platform.system() == "Darwin" and platform.machine() == "arm64"


def create_backends(
  model_cache_dir: Path | None = None,
  *,
  qwen_asr_python: Path | None = None,
  runtime_dir: Path | None = None,
  torch_compile: bool = False,
  qwen_engine: str = "nari",
  nari_config: NariRuntimeConfig | None = None,
) -> BackendBundle:
  """Create MLX on supported Macs and PyTorch everywhere else."""
  if is_apple_silicon() and importlib.util.find_spec("mlx_audio") is not None:
    from app.backends.mlx import MLXSTTBackend, MLXTTSBackend
    from app.backends.mlx.worker import MLXWorker

    worker = MLXWorker()

    details = DeviceInfo(device_name="Apple GPU", runtime_version="MLX")
    log_runtime_summary(
      backend="mlx",
      device="metal",
      details=details,
    )
    return BackendBundle(
      name="mlx",
      device="metal",
      tts=MLXTTSBackend(model_cache_dir, worker=worker),
      stt=STTRegistry((MLXSTTBackend(model_cache_dir, worker=worker),)),
      device_info=details,
    )

  import torch

  from app.backends.pytorch import PyTorchSTTBackend, PyTorchTTSBackend
  from app.backends.pytorch.device import select_device
  from app.backends.pytorch.nemo_stt import NeMoSTTBackend
  from app.backends.pytorch.qwen_asr import QwenASRSTTBackend

  device = str(select_device(torch))
  details = inspect_pytorch_device(torch, device)
  log_runtime_summary(
    backend="pytorch",
    device=device,
    details=details,
    torch=torch,
  )
  return BackendBundle(
    name="pytorch",
    device=device,
    tts=PyTorchTTSBackend(
      model_cache_dir,
      compile_model=torch_compile,
      engine=qwen_engine,
      nari_config=nari_config,
    ),
    stt=STTRegistry(
      (
        PyTorchSTTBackend(model_cache_dir),
        QwenASRSTTBackend(
          model_cache_dir,
          python_path=qwen_asr_python,
          managed_runtime_dir=runtime_dir,
          device=device,
        ),
        NeMoSTTBackend(
          model_cache_dir,
          managed_runtime_dir=runtime_dir,
          device=device,
        ),
      )
    ),
    device_info=details,
  )
