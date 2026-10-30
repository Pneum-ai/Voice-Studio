"""Exercise the installed MLX loader with a small local Whisper checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.backends.mlx.stt import MLXSTTBackend


def test_whisper_loads_from_a_custom_cache_snapshot(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Load real MLX weights from a hash path without relying on a hub directory."""
  mx = pytest.importorskip("mlx.core")
  whisper = pytest.importorskip("mlx_audio.stt.models.whisper.whisper")
  utils = pytest.importorskip("mlx.utils")
  snapshot = tmp_path / "custom-cache" / "snapshots" / ("a" * 40)
  snapshot.mkdir(parents=True)
  dimensions = {
    "n_mels": 80,
    "n_audio_ctx": 4,
    "n_audio_state": 8,
    "n_audio_head": 2,
    "n_audio_layer": 1,
    "n_vocab": 16,
    "n_text_ctx": 4,
    "n_text_state": 8,
    "n_text_head": 2,
    "n_text_layer": 1,
  }
  (snapshot / "config.json").write_text(
    json.dumps({**dimensions, "model_type": "whisper"}), encoding="utf-8"
  )
  original = whisper.Model(whisper.ModelDimensions(**dimensions))
  weights = dict(utils.tree_flatten(original.parameters()))
  mx.save_safetensors(str(snapshot / "weights.safetensors"), weights)
  monkeypatch.setattr("app.backends.mlx.stt.repository_snapshot_path", lambda *_: snapshot)

  loaded = MLXSTTBackend(tmp_path)._load_sync("whisper-base")

  assert isinstance(loaded, whisper.Model)
  assert not loaded.training
  for name, value in utils.tree_flatten(loaded.parameters()):
    assert mx.array_equal(value, weights[name].astype(value.dtype))
