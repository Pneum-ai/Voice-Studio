"""Safe hashing, path resolution, and recoverable file replacement helpers."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from app.core.errors import ValidationError


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
  """Return a file SHA-256 digest without loading the whole file into memory."""
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    while chunk := stream.read(chunk_size):
      digest.update(chunk)
  return digest.hexdigest()


def resolve_data_path(data_dir: Path, relative_path: str) -> Path:
  """Resolve a stored relative path and reject traversal outside data storage."""
  if Path(relative_path).is_absolute():
    raise ValidationError("Stored data path must be relative.", "invalid_data_path")
  root = data_dir.resolve()
  resolved = (root / relative_path).resolve()
  try:
    resolved.relative_to(root)
  except ValueError as error:
    raise ValidationError("Stored data path escapes the data directory.") from error
  return resolved


def promote_file(staged: Path, destination: Path) -> None:
  """Atomically replace a destination with a fully written staged file."""
  destination.parent.mkdir(parents=True, exist_ok=True)
  os.replace(staged, destination)


def unlink_if_exists(path: Path) -> bool:
  """Delete one file if present and report whether it was removed."""
  try:
    path.unlink()
    return True
  except FileNotFoundError:
    return False
