"""Isolated Qwen3-ASR adapter for the Qwen TTS dependency boundary."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path

from app.backends.cache import (
  cache_root,
  delete_repository,
  download_repository,
  is_repository_cached,
  repository_size,
  repository_snapshot_path,
)
from app.backends.protocols import (
  DownloadProgressCallback,
  TranscriptionRequest,
  TranscriptionResult,
)
from app.core.errors import ValidationError

MODEL_REPOSITORIES = {
  "qwen3-asr-0.6B": "Qwen/Qwen3-ASR-0.6B",
  "qwen3-asr-1.7B": "Qwen/Qwen3-ASR-1.7B",
}
REQUIRED_SNAPSHOT_FILES = (
  "chat_template.json",
  "config.json",
  "preprocessor_config.json",
  "tokenizer_config.json",
)
RESPONSE_PREFIX = "VOICE_ASR:"


class QwenASRSTTBackend:
  """Run Qwen3-ASR in a persistent, dependency-isolated Python worker."""

  def __init__(
    self,
    model_cache_dir: Path | None = None,
    *,
    python_path: Path | None = None,
    managed_runtime_dir: Path | None = None,
    device: str = "cpu",
  ) -> None:
    """Resolve the optional worker interpreter and initialize an unloaded adapter."""
    self._cache_root = cache_root(model_cache_dir)
    self._configured_python = python_path
    self._managed_runtime_dir = managed_runtime_dir
    self._device = device
    self._process: asyncio.subprocess.Process | None = None
    self._loaded_model: str | None = None
    self._lock = asyncio.Lock()

  def _resolve_python(self) -> Path | None:
    """Use an explicit Qwen environment or the current compatible environment."""
    if self._configured_python is not None:
      configured = self._configured_python.expanduser().absolute()
      return configured if configured.is_file() else None
    if self._managed_runtime_dir is not None:
      executable = "python.exe" if sys.platform == "win32" else "python"
      scripts = "Scripts" if sys.platform == "win32" else "bin"
      managed = self._managed_runtime_dir / "qwen-asr" / scripts / executable
      if managed.is_file():
        # Resolving bin/python follows the venv symlink to its base interpreter,
        # which bypasses pyvenv.cfg and hides the isolated dependencies.
        return managed.absolute()
    if importlib.util.find_spec("qwen_asr") is not None:
      return Path(sys.executable)
    return None

  @property
  def loaded_model(self) -> str | None:
    """Return the canonical Qwen3-ASR model held by the worker."""
    return self._loaded_model

  def supports(self, model_id: str) -> bool:
    """Return whether the model and isolated Qwen runtime are available."""
    return model_id in MODEL_REPOSITORIES and self._resolve_python() is not None

  def _repository(self, model_id: str) -> str:
    """Resolve a canonical model ID to its Hugging Face repository."""
    try:
      return MODEL_REPOSITORIES[model_id]
    except KeyError as error:
      raise ValidationError(f"Unsupported Qwen3-ASR model: {model_id}") from error

  def is_cached(self, model_id: str) -> bool:
    """Return whether complete Qwen3-ASR weights are cached."""
    repository = self._repository(model_id)
    if not is_repository_cached(self._cache_root, repository):
      return False
    try:
      snapshot = repository_snapshot_path(self._cache_root, repository)
    except FileNotFoundError:
      return False
    return all((snapshot / filename).is_file() for filename in REQUIRED_SNAPSHOT_FILES)

  def cache_size(self, model_id: str) -> int | None:
    """Return cached Qwen3-ASR bytes."""
    return repository_size(self._cache_root, self._repository(model_id))

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Download Qwen3-ASR weights without starting the worker and report progress."""
    await download_repository(self._cache_root, self._repository(model_id), progress)

  async def _start_worker(self) -> None:
    """Start the configured interpreter and local JSON-line worker."""
    if self._process is not None and self._process.returncode is None:
      return
    python_path = self._resolve_python()
    if python_path is None:
      raise RuntimeError("Qwen3-ASR runtime is not configured.")
    worker_path = Path(__file__).with_name("qwen_asr_worker.py")
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    self._process = await asyncio.create_subprocess_exec(
      str(python_path),
      str(worker_path),
      stdin=asyncio.subprocess.PIPE,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.DEVNULL,
      env=environment,
    )

  async def _command(self, payload: dict[str, object]) -> dict[str, object]:
    """Send one worker command and return its prefixed JSON response."""
    await self._start_worker()
    process = self._process
    if process is None or process.stdin is None or process.stdout is None:
      raise RuntimeError("Qwen3-ASR worker did not expose communication pipes.")
    process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    await process.stdin.drain()
    while True:
      line = await process.stdout.readline()
      if not line:
        raise RuntimeError("Qwen3-ASR worker exited before returning a response.")
      decoded = line.decode("utf-8", errors="replace").strip()
      if not decoded.startswith(RESPONSE_PREFIX):
        continue
      response = json.loads(decoded.removeprefix(RESPONSE_PREFIX))
      if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "Qwen3-ASR worker failed.")))
      return response

  async def load(self, model_id: str) -> None:
    """Download when needed and load Qwen3-ASR in its isolated worker."""
    if not self.supports(model_id):
      raise ValidationError(f"Qwen3-ASR runtime is unavailable for {model_id}.")
    async with self._lock:
      if self._loaded_model == model_id and self._process is not None:
        return
      await self._stop_worker()
      if not await asyncio.to_thread(self.is_cached, model_id):
        await self.download(model_id)
      if not await asyncio.to_thread(self.is_cached, model_id):
        raise RuntimeError("The Qwen3-ASR model snapshot is incomplete after downloading.")
      snapshot = repository_snapshot_path(self._cache_root, self._repository(model_id))
      await self._command({"command": "load", "model_path": str(snapshot), "device": self._device})
      self._loaded_model = model_id

  async def unload(self) -> None:
    """Stop the worker and release its model memory."""
    async with self._lock:
      await self._stop_worker()

  async def _stop_worker(self) -> None:
    """Terminate the worker while its lifecycle lock is held."""
    process = self._process
    self._process = None
    self._loaded_model = None
    if process is None or process.returncode is not None:
      return
    process.terminate()
    try:
      await asyncio.wait_for(process.wait(), timeout=5)
    except TimeoutError:
      process.kill()
      await process.wait()

  async def delete(self, model_id: str) -> None:
    """Unload the selected Qwen3-ASR model and remove its weights."""
    if self._loaded_model == model_id:
      await self.unload()
    await delete_repository(self._cache_root, self._repository(model_id))

  async def transcribe(
    self,
    request: TranscriptionRequest,
  ) -> TranscriptionResult:
    """Transcribe audio through the persistent isolated worker."""
    async with self._lock:
      response = await self._command(
        {
          "command": "transcribe",
          "audio_path": str(request.audio_path),
          "language": request.language,
          "context": request.context,
        }
      )
    return TranscriptionResult(
      text=str(response.get("text", "")),
      language=str(response["language"]) if response.get("language") else None,
    )
