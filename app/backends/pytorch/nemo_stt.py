"""Optional NVIDIA NeMo adapter for Canary and Parakeet transcription."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

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
from app.utils.async_work import run_in_daemon_thread

MODEL_REPOSITORIES = {
  "nvidia-canary-1b-v2": "nvidia/canary-1b-v2",
  "nvidia-parakeet-tdt-0.6b-v3": "nvidia/parakeet-tdt-0.6b-v3",
}
RESPONSE_PREFIX = "VOICE_NEMO_ASR:"


class NeMoSTTBackend:
  """Run NVIDIA Canary and Parakeet checkpoints through optional NeMo."""

  def __init__(
    self,
    model_cache_dir: Path | None = None,
    *,
    managed_runtime_dir: Path | None = None,
    device: str = "cpu",
    enabled: bool = True,
  ) -> None:
    """Initialize an unloaded adapter and detect the optional NeMo runtime."""
    self._cache_root = cache_root(model_cache_dir)
    self._managed_runtime_dir = managed_runtime_dir
    self._device = device
    self._enabled = enabled
    self._model: Any | None = None
    self._loaded_model: str | None = None
    self._process: asyncio.subprocess.Process | None = None
    self._lock = asyncio.Lock()

  def _resolve_python(self) -> Path | None:
    """Prefer the managed NeMo worker and otherwise use an in-process runtime."""
    if self._managed_runtime_dir is not None:
      executable = "python.exe" if sys.platform == "win32" else "python"
      scripts = "Scripts" if sys.platform == "win32" else "bin"
      managed = self._managed_runtime_dir / "nvidia-asr" / scripts / executable
      if managed.is_file():
        # Preserve the venv launcher path so Python discovers the worker's site-packages.
        return managed.absolute()
    if importlib.util.find_spec("nemo") is not None:
      return Path(sys.executable).resolve()
    return None

  def _uses_worker(self) -> bool:
    """Return whether NeMo lives outside the active server interpreter."""
    python_path = self._resolve_python()
    return python_path is not None and python_path != Path(sys.executable).resolve()

  @property
  def loaded_model(self) -> str | None:
    """Return the canonical NVIDIA model currently in memory."""
    return self._loaded_model

  def supports(self, model_id: str) -> bool:
    """Return whether NeMo is installed and owns the selected model."""
    available = self._enabled and self._resolve_python() is not None
    return available and model_id in MODEL_REPOSITORIES

  def _repository(self, model_id: str) -> str:
    """Resolve a canonical model ID to its Hugging Face repository."""
    try:
      return MODEL_REPOSITORIES[model_id]
    except KeyError as error:
      raise ValidationError(f"Unsupported NVIDIA transcription model: {model_id}") from error

  def is_cached(self, model_id: str) -> bool:
    """Return whether complete NVIDIA model weights are cached."""
    return is_repository_cached(self._cache_root, self._repository(model_id))

  def cache_size(self, model_id: str) -> int | None:
    """Return cached NVIDIA model bytes."""
    return repository_size(self._cache_root, self._repository(model_id))

  async def download(
    self,
    model_id: str,
    progress: DownloadProgressCallback | None = None,
  ) -> None:
    """Download NVIDIA model weights without loading them and report byte progress."""
    await download_repository(self._cache_root, self._repository(model_id), progress)

  async def load(self, model_id: str) -> None:
    """Download when necessary and load a local NeMo checkpoint."""
    if not self.supports(model_id):
      raise ValidationError(f"The NeMo runtime is unavailable for {model_id}.")
    async with self._lock:
      worker_loaded = self._process is not None and self._process.returncode is None
      if self._loaded_model == model_id and (self._model is not None or worker_loaded):
        return
      await self._unload_locked()
      if not await asyncio.to_thread(self.is_cached, model_id):
        await self.download(model_id)
      if self._uses_worker():
        snapshot = repository_snapshot_path(self._cache_root, self._repository(model_id))
        await self._command(
          {
            "command": "load",
            "model_id": model_id,
            "model_path": str(snapshot),
            "device": self._device,
          }
        )
      else:
        self._model = await run_in_daemon_thread(
          self._load_sync,
          model_id,
          thread_name="voice-nemo-stt-load",
        )
      self._loaded_model = model_id

  async def _start_worker(self) -> None:
    """Start the managed NeMo interpreter and local JSON-line worker."""
    if self._process is not None and self._process.returncode is None:
      return
    python_path = self._resolve_python()
    if python_path is None:
      raise RuntimeError("NVIDIA ASR runtime is not configured.")
    worker_path = Path(__file__).with_name("nemo_stt_worker.py")
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
      raise RuntimeError("NVIDIA ASR worker did not expose communication pipes.")
    process.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    await process.stdin.drain()
    while True:
      line = await process.stdout.readline()
      if not line:
        raise RuntimeError("NVIDIA ASR worker exited before returning a response.")
      decoded = line.decode("utf-8", errors="replace").strip()
      if not decoded.startswith(RESPONSE_PREFIX):
        continue
      response = json.loads(decoded.removeprefix(RESPONSE_PREFIX))
      if not response.get("ok"):
        raise RuntimeError(str(response.get("error", "NVIDIA ASR worker failed.")))
      return response

  def _load_sync(self, model_id: str) -> Any:
    """Restore the repository's packaged NeMo checkpoint."""
    from nemo.collections.asr.models import ASRModel  # pyright: ignore[reportMissingImports]

    snapshot = repository_snapshot_path(self._cache_root, self._repository(model_id))
    checkpoints = sorted(snapshot.glob("*.nemo"))
    if not checkpoints:
      checkpoints = sorted(snapshot.rglob("*.nemo"))
    if not checkpoints:
      raise RuntimeError(f"No .nemo checkpoint was found in {snapshot}.")
    model = ASRModel.restore_from(str(checkpoints[0]), map_location=self._device)
    model = model.to(self._device)
    model.eval()
    if model_id.startswith("nvidia-parakeet") and hasattr(model, "change_attention_model"):
      model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[256, 256],
      )
    return model

  async def unload(self) -> None:
    """Release the loaded NeMo model and accelerator cache."""
    async with self._lock:
      await self._unload_locked()

  async def _unload_locked(self) -> None:
    """Release NeMo state while the lifecycle lock is held."""
    process = self._process
    self._process = None
    self._model = None
    self._loaded_model = None
    if process is not None and process.returncode is None:
      process.terminate()
      try:
        await asyncio.wait_for(process.wait(), timeout=5)
      except TimeoutError:
        process.kill()
        await process.wait()
    try:
      import torch

      if torch.cuda.is_available():
        torch.cuda.empty_cache()
    except ImportError:
      pass

  async def delete(self, model_id: str) -> None:
    """Unload the selected NVIDIA model and remove its weights."""
    if self._loaded_model == model_id:
      await self.unload()
    await delete_repository(self._cache_root, self._repository(model_id))

  async def transcribe(
    self,
    request: TranscriptionRequest,
  ) -> TranscriptionResult:
    """Run blocking NeMo recognition outside the event loop."""
    if self._loaded_model is None:
      raise RuntimeError("NVIDIA transcription model is not loaded.")
    if self._uses_worker():
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
    if self._model is None:
      raise RuntimeError("NVIDIA transcription model is not loaded.")
    return await run_in_daemon_thread(
      self._transcribe_sync,
      request,
      thread_name="voice-nemo-stt-transcribe",
    )

  def _transcribe_sync(self, request: TranscriptionRequest) -> TranscriptionResult:
    """Translate generic recognition controls into NeMo model arguments."""
    model = self._model
    model_id = self._loaded_model
    if model is None or model_id is None:
      raise RuntimeError("NVIDIA transcription model is not loaded.")
    kwargs: dict[str, Any] = {}
    if model_id.startswith("nvidia-canary"):
      if not request.language:
        raise ValidationError("Canary requires a source language.", "language_required")
      kwargs.update(
        source_lang=request.language,
        target_lang=request.language,
      )
      parameters = inspect.signature(model.transcribe).parameters
      accepts_keywords = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
      )
      if request.context:
        if "decodercontext" not in parameters and not accepts_keywords:
          raise ValidationError(
            "The installed NeMo runtime does not expose Canary context.",
            "unsupported_context",
          )
        kwargs["decodercontext"] = request.context
    outputs = model.transcribe([str(request.audio_path)], **kwargs)
    output = outputs[0]
    text = str(getattr(output, "text", output)).strip()
    detected = getattr(output, "language", None) or request.language
    return TranscriptionResult(
      text=text,
      language=str(detected) if detected else None,
    )
