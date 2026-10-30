"""Hugging Face cache layout and completeness tests."""

from __future__ import annotations

from pathlib import Path

from app.backends.cache import (
  is_repository_cached,
  repository_downloaded_size,
  repository_size,
)


def create_materialized_snapshot(root: Path, repository_id: str) -> Path:
  """Create the cache shape used when snapshot files are not blob symlinks."""
  repository = root / f"models--{repository_id.replace('/', '--')}"
  revision = "0123456789abcdef"
  (repository / "refs").mkdir(parents=True)
  (repository / "refs" / "main").write_text(revision, encoding="utf-8")
  snapshot = repository / "snapshots" / revision
  snapshot.mkdir(parents=True)
  return snapshot


def test_materialized_snapshot_reports_complete_size(tmp_path: Path) -> None:
  """Count direct snapshot files when a cache has no blob objects."""
  repository_id = "example/model"
  snapshot = create_materialized_snapshot(tmp_path, repository_id)
  (snapshot / "config.json").write_bytes(b"{}")
  (snapshot / "model.safetensors").write_bytes(b"weights")
  assert is_repository_cached(tmp_path, repository_id)
  assert repository_size(tmp_path, repository_id) == 9


def test_incomplete_blob_marks_snapshot_unavailable(tmp_path: Path) -> None:
  """Never advertise a repository while an incomplete blob remains."""
  repository_id = "example/model"
  snapshot = create_materialized_snapshot(tmp_path, repository_id)
  (snapshot / "model.safetensors").write_bytes(b"weights")
  blobs = tmp_path / "models--example--model" / "blobs"
  blobs.mkdir()
  (blobs / "pending.incomplete").write_bytes(b"partial")
  assert not is_repository_cached(tmp_path, repository_id)


def test_active_download_size_includes_complete_and_partial_blobs(tmp_path: Path) -> None:
  """Report byte growth while Hugging Face retains an incomplete weight blob."""
  repository_id = "example/model"
  blobs = tmp_path / "models--example--model" / "blobs"
  blobs.mkdir(parents=True)
  (blobs / "metadata").write_bytes(b"123")
  (blobs / "weights.incomplete").write_bytes(b"1234567")

  assert repository_downloaded_size(tmp_path, repository_id) == 10


def test_nemo_checkpoint_marks_nvidia_repository_complete(tmp_path: Path) -> None:
  """Recognize packaged NeMo checkpoints as downloaded model weights."""
  repository_id = "nvidia/canary"
  snapshot = create_materialized_snapshot(tmp_path, repository_id)
  (snapshot / "canary.nemo").write_bytes(b"weights")

  assert is_repository_cached(tmp_path, repository_id)
