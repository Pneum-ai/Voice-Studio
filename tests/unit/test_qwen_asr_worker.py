"""Qwen3-ASR isolated worker import-boundary tests."""

from __future__ import annotations

import sys
from pathlib import Path

from app.backends.pytorch import qwen_asr_worker


def test_worker_adapter_directory_cannot_shadow_qwen_package(monkeypatch) -> None:
  """Remove the script directory that otherwise resolves the local adapter first."""
  worker_directory = Path(qwen_asr_worker.__file__).resolve().parent
  retained = worker_directory.parent
  monkeypatch.setattr(sys, "path", [str(worker_directory), str(retained)])

  qwen_asr_worker._remove_worker_import_shadow()

  assert sys.path == [str(retained)]
