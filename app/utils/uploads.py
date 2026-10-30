"""Bounded temporary upload handling for audio routes."""

from __future__ import annotations

import asyncio
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import UploadFile

from app.core.errors import ValidationError
from app.utils.files import unlink_if_exists

UPLOAD_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def temporary_upload(
  upload: UploadFile,
  *,
  max_bytes: int,
) -> AsyncIterator[Path]:
  """Stream an upload to a bounded temporary path and always remove it."""
  suffix = _safe_suffix(upload.filename)
  handle = tempfile.NamedTemporaryFile(prefix="local-voice-", suffix=suffix, delete=False)
  path = Path(handle.name)
  size = 0
  try:
    while chunk := await upload.read(UPLOAD_CHUNK_BYTES):
      size += len(chunk)
      if size > max_bytes:
        raise ValidationError(
          f"Upload exceeds the {max_bytes}-byte limit.",
          "upload_too_large",
        )
      await asyncio.to_thread(handle.write, chunk)
    await asyncio.to_thread(handle.flush)
    handle.close()
    if size == 0:
      raise ValidationError("Uploaded file is empty.", "empty_upload")
    yield path
  finally:
    if not handle.closed:
      handle.close()
    await upload.close()
    await asyncio.to_thread(unlink_if_exists, path)


def _safe_suffix(filename: str | None) -> str:
  """Return a short alphanumeric suffix without trusting upload paths."""
  suffix = Path(filename or "").suffix.casefold()
  if 1 < len(suffix) <= 10 and suffix[1:].isalnum():
    return suffix
  return ".upload"
