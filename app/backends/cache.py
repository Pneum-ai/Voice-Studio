"""Hugging Face cache inspection and deletion helpers."""

from __future__ import annotations

import asyncio
import logging
import shutil
from pathlib import Path

from huggingface_hub import HfApi, constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from app.backends.protocols import DownloadProgressCallback
from app.utils.async_work import run_in_daemon_thread

LOGGER = logging.getLogger(__name__)
DOWNLOAD_PROGRESS_INTERVAL_SECONDS = 0.5
WEIGHT_SUFFIXES = (".bin", ".nemo", ".npz", ".pt", ".pth", ".safetensors")


def cache_root(configured: Path | None) -> Path:
  """Return the configured or platform-default Hugging Face cache root."""
  return configured or Path(constants.HF_HUB_CACHE)


def repository_cache_path(root: Path, repository_id: str) -> Path:
  """Return the local cache directory for a Hugging Face repository."""
  return root / f"models--{repository_id.replace('/', '--')}"


def is_repository_cached(root: Path, repository_id: str) -> bool:
  """Return whether complete model weight files exist without partial blobs."""
  repository = repository_cache_path(root, repository_id)
  if not repository.is_dir():
    return False
  blobs = repository / "blobs"
  if blobs.is_dir() and any(blobs.glob("*.incomplete")):
    return False
  try:
    snapshot = repository_snapshot_path(root, repository_id)
  except FileNotFoundError:
    return False
  return any(
    path.is_file() and path.suffix.lower() in WEIGHT_SUFFIXES for path in snapshot.rglob("*")
  )


def repository_snapshot_path(root: Path, repository_id: str) -> Path:
  """Resolve a complete cached snapshot without contacting the model host."""
  try:
    snapshot = snapshot_download(
      repo_id=repository_id,
      cache_dir=str(root),
      local_files_only=True,
    )
  except LocalEntryNotFoundError as error:
    raise FileNotFoundError(f"No complete local snapshot exists for {repository_id}.") from error
  return Path(snapshot)


def repository_size(root: Path, repository_id: str) -> int | None:
  """Return cached bytes for blob-linked or materialized snapshot layouts."""
  repository = repository_cache_path(root, repository_id)
  blobs = repository / "blobs"
  try:
    blob_files = (
      [path for path in blobs.iterdir() if path.is_file() and not path.name.endswith(".incomplete")]
      if blobs.is_dir()
      else []
    )
    if blob_files:
      return sum(path.stat().st_size for path in blob_files)
    snapshot = repository_snapshot_path(root, repository_id)
    return sum(path.stat().st_size for path in snapshot.rglob("*") if path.is_file())
  except FileNotFoundError:
    return None
  except OSError:
    return None


def repository_downloaded_size(root: Path, repository_id: str) -> int:
  """Return completed and partial blob bytes for an active repository download."""
  blobs = repository_cache_path(root, repository_id) / "blobs"
  try:
    return sum(path.stat().st_size for path in blobs.iterdir() if path.is_file())
  except (FileNotFoundError, OSError):
    return 0


def repository_remote_size(repository_id: str) -> int | None:
  """Return repository bytes advertised by Hugging Face when metadata is available."""
  try:
    information = HfApi().model_info(repository_id, files_metadata=True)
    sizes = [file.size for file in information.siblings or () if file.size is not None]
    return sum(sizes) if sizes else None
  except Exception as error:
    LOGGER.warning("Could not read download size for %s: %s", repository_id, error)
    return None


async def download_repository(
  root: Path,
  repository_id: str,
  progress: DownloadProgressCallback | None = None,
) -> None:
  """Download a model snapshot while optionally reporting byte-level progress."""
  root.mkdir(parents=True, exist_ok=True)
  total = (
    await run_in_daemon_thread(
      repository_remote_size,
      repository_id,
      thread_name="voice-model-size",
    )
    if progress is not None
    else None
  )
  download = asyncio.create_task(
    run_in_daemon_thread(
      snapshot_download,
      repo_id=repository_id,
      cache_dir=str(root),
      thread_name="voice-model-download",
    )
  )
  while not download.done():
    if progress is not None:
      current = await asyncio.to_thread(repository_downloaded_size, root, repository_id)
      await progress(current, total)
    try:
      await asyncio.wait_for(
        asyncio.shield(download),
        timeout=DOWNLOAD_PROGRESS_INTERVAL_SECONDS,
      )
    except TimeoutError:
      continue
  await download
  if progress is not None:
    current = await asyncio.to_thread(repository_downloaded_size, root, repository_id)
    await progress(current, total or current)


async def delete_repository(root: Path, repository_id: str) -> None:
  """Delete exactly one resolved repository cache outside the event loop."""
  repository = repository_cache_path(root, repository_id).resolve()
  resolved_root = root.resolve()
  repository.relative_to(resolved_root)
  if repository.is_dir():
    await run_in_daemon_thread(
      shutil.rmtree,
      repository,
      thread_name="voice-model-delete",
    )
