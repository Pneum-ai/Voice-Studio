"""Qwen3-ASR model-cache boundary tests."""

from __future__ import annotations

from pathlib import Path

from app.backends.pytorch import qwen_asr
from app.backends.pytorch.qwen_asr import (
  REQUIRED_SNAPSHOT_FILES,
  QwenASRSTTBackend,
)


def test_qwen_cache_requires_processor_chat_template(tmp_path: Path, monkeypatch) -> None:
  """Do not load weights before every processor configuration file exists."""
  snapshot = tmp_path / "snapshot"
  snapshot.mkdir()
  (snapshot / "model.safetensors").write_bytes(b"weights")
  for filename in REQUIRED_SNAPSHOT_FILES:
    if filename != "chat_template.json":
      (snapshot / filename).write_text("{}", encoding="utf-8")
  monkeypatch.setattr(qwen_asr, "is_repository_cached", lambda *_: True)
  monkeypatch.setattr(qwen_asr, "repository_snapshot_path", lambda *_: snapshot)
  backend = QwenASRSTTBackend(model_cache_dir=tmp_path)

  assert not backend.is_cached("qwen3-asr-0.6B")

  (snapshot / "chat_template.json").write_text("{}", encoding="utf-8")
  assert backend.is_cached("qwen3-asr-0.6B")
