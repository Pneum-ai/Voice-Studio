"""Managed optional model-runtime installation tests."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from app.backends.pytorch.nemo_stt import NeMoSTTBackend
from app.backends.pytorch.qwen_asr import QwenASRSTTBackend
from app.core.config import Settings
from app.services.jobs import JobService
from app.services.model_runtimes import ModelRuntimeService


@pytest.mark.asyncio
async def test_ensure_all_skips_an_existing_matching_installation(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Avoid package resolution on later launches when verified runtimes are unchanged."""
  settings = Settings(data_dir=tmp_path, prewarm=False)
  service = ModelRuntimeService(settings, JobService(300))
  monkeypatch.setattr(service, "_cuda_torch_requirement", lambda: None)
  for name in ("qwen-asr", "nvidia-asr"):
    python_path = service._runtime_python(settings.runtime_dir / name)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.touch()
  service._marker_path.write_text(service._runtime_fingerprint(), encoding="utf-8")
  installer = AsyncMock()
  monkeypatch.setattr(service, "install_all", installer)

  await service.ensure_all()

  installer.assert_not_awaited()


@pytest.mark.skipif(sys.platform == "win32", reason="Windows venv launchers are not symlinks")
@pytest.mark.parametrize(
  ("backend_type", "runtime_name"),
  (
    (QwenASRSTTBackend, "qwen-asr"),
    (NeMoSTTBackend, "nvidia-asr"),
  ),
)
def test_managed_runtime_preserves_virtual_environment_python_symlink(
  tmp_path: Path,
  backend_type: type[QwenASRSTTBackend] | type[NeMoSTTBackend],
  runtime_name: str,
) -> None:
  """Avoid resolving a venv launcher into the dependency-free base interpreter."""
  base_python = tmp_path / "base-python"
  base_python.touch()
  launcher = tmp_path / runtime_name / "bin" / "python"
  launcher.parent.mkdir(parents=True)
  launcher.symlink_to(base_python)
  backend = backend_type(managed_runtime_dir=tmp_path)

  assert backend._resolve_python() == launcher.absolute()


def test_cuda_worker_requirement_matches_the_active_torch(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Use the active CUDA family for the isolated NVIDIA worker."""
  import torch

  monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
  monkeypatch.setattr(torch.version, "cuda", "13.0")
  monkeypatch.setattr(torch, "__version__", "2.13.0+cu130")

  assert ModelRuntimeService._cuda_torch_requirement() == (
    "https://download.pytorch.org/whl/cu130",
    "torch==2.13.0+cu130",
  )
