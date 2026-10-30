"""Background installation of optional isolated transcription runtimes."""

from __future__ import annotations

import asyncio
import logging
import shutil
import sys
from pathlib import Path

from app.core.config import Settings
from app.core.errors import ConflictError
from app.domain.jobs import Job, JobState
from app.services.jobs import JobService

LOGGER = logging.getLogger(__name__)


class ModelRuntimeService:
  """Install optional model dependencies without changing request handlers."""

  def __init__(
    self,
    settings: Settings,
    jobs: JobService,
    *,
    auto_install: bool = True,
  ) -> None:
    """Store runtime paths and retained job tracking."""
    self.settings = settings
    self.jobs = jobs
    self.auto_install = auto_install

  async def ensure_all(self) -> None:
    """Install required isolated runtimes before the application starts serving."""
    if not self.auto_install:
      return
    if await asyncio.to_thread(self._installation_is_current):
      LOGGER.info("Managed transcription runtimes are ready.")
      return
    LOGGER.info("Installing managed transcription runtimes for this environment.")
    job = await self.install_all()
    async for snapshot in self.jobs.subscribe(job.id):
      if not snapshot.terminal:
        continue
      if snapshot.state is JobState.COMPLETE:
        LOGGER.info("Managed transcription runtimes installed successfully.")
        return
      detail = snapshot.error or f"installation {snapshot.state}"
      raise RuntimeError(f"Managed transcription runtime installation failed: {detail}")

  async def install_all(self) -> Job:
    """Install every compatible optional transcription runtime in the background."""
    resource_id = "transcription-runtimes"
    if await self.jobs.active_job_for(resource_id):
      raise ConflictError("Transcription runtimes are already being installed.")

    async def install(job_id: str) -> None:
      """Create dependency-isolated Qwen3-ASR and NVIDIA ASR workers."""
      uv = shutil.which("uv")
      if uv is None:
        raise RuntimeError("uv was not found on PATH; install uv before adding runtimes.")
      self.settings.runtime_dir.mkdir(parents=True, exist_ok=True)
      qwen_dir = self.settings.runtime_dir / "qwen-asr"
      await self.jobs.update(job_id, progress=0.1, stage="creating Qwen3-ASR runtime")
      await self._run(uv, "venv", "--allow-existing", str(qwen_dir))
      qwen_python = self._runtime_python(qwen_dir)
      await self.jobs.update(job_id, progress=0.25, stage="installing Qwen3-ASR runtime")
      await self._run(
        uv,
        "pip",
        "install",
        "--python",
        str(qwen_python),
        "--no-progress",
        "qwen-asr==0.0.6",
      )
      nvidia_dir = self.settings.runtime_dir / "nvidia-asr"
      await self.jobs.update(job_id, progress=0.55, stage="creating NVIDIA ASR runtime")
      await self._run(
        uv,
        "venv",
        "--allow-existing",
        "--python",
        "3.12",
        str(nvidia_dir),
      )
      nvidia_python = self._runtime_python(nvidia_dir)
      await self.jobs.update(job_id, progress=0.62, stage="installing NVIDIA ASR runtime")
      await self._run(
        uv,
        "pip",
        "install",
        "--python",
        str(nvidia_python),
        "--no-progress",
        "nemo-toolkit[asr]==2.7.3",
        "kaldialign==0.9.1",
        "numba>=0.63,<0.64",
      )
      cuda_torch = self._cuda_torch_requirement()
      if cuda_torch is not None:
        index_url, torch_requirement = cuda_torch
        await self.jobs.update(
          job_id,
          progress=0.88,
          stage="installing NVIDIA accelerator runtime",
        )
        await self._run(
          uv,
          "pip",
          "install",
          "--python",
          str(nvidia_python),
          "--no-progress",
          "--no-deps",
          "--index",
          index_url,
          torch_requirement,
        )
      await self.jobs.update(job_id, progress=0.95, stage="verifying runtimes")
      await self._run(str(qwen_python), "-c", "import qwen_asr")
      await self._run(str(nvidia_python), "-c", "import nemo.collections.asr")
      await asyncio.to_thread(
        self._marker_path.write_text,
        self._runtime_fingerprint(),
        encoding="utf-8",
      )

    return await self.jobs.start(
      kind="runtime_install",
      resource_id=resource_id,
      work=install,
    )

  @property
  def _marker_path(self) -> Path:
    """Return the manifest written only after both runtimes verify successfully."""
    return self.settings.runtime_dir / ".voice-open-asr-runtimes"

  def _runtime_fingerprint(self) -> str:
    """Describe the exact managed dependency set needed by this server."""
    requirements = [
      "qwen-asr==0.0.6",
      "nemo-toolkit[asr]==2.7.3",
      "kaldialign==0.9.1",
      "numba>=0.63,<0.64",
    ]
    cuda_torch = self._cuda_torch_requirement()
    if cuda_torch is not None:
      requirements.extend(cuda_torch)
    return "\n".join((sys.platform, *requirements, ""))

  def _installation_is_current(self) -> bool:
    """Return whether verified environments match the current dependency manifest."""
    qwen_python = self._runtime_python(self.settings.runtime_dir / "qwen-asr")
    nvidia_python = self._runtime_python(self.settings.runtime_dir / "nvidia-asr")
    if not qwen_python.is_file() or not nvidia_python.is_file():
      return False
    try:
      manifest = self._marker_path.read_text(encoding="utf-8")
    except OSError:
      return False
    return manifest == self._runtime_fingerprint()

  @staticmethod
  def _runtime_python(runtime_dir: Path) -> Path:
    """Return the platform-specific interpreter inside a managed environment."""
    if sys.platform == "win32":
      return runtime_dir / "Scripts" / "python.exe"
    return runtime_dir / "bin" / "python"

  @staticmethod
  def _cuda_torch_requirement() -> tuple[str, str] | None:
    """Mirror the active CUDA PyTorch build inside the NVIDIA worker."""
    try:
      import torch
    except ImportError:
      return None
    cuda_version = torch.version.cuda
    if not cuda_version or not torch.cuda.is_available():
      return None
    cuda_tag = "cu" + cuda_version.replace(".", "")
    torch_version = torch.__version__.split("+", maxsplit=1)[0]
    index_url = f"https://download.pytorch.org/whl/{cuda_tag}"
    return index_url, f"torch=={torch_version}+{cuda_tag}"

  @staticmethod
  async def _run(*command: str) -> None:
    """Run one installer command and preserve concise failure context."""
    process = await asyncio.create_subprocess_exec(
      *command,
      stdout=asyncio.subprocess.PIPE,
      stderr=asyncio.subprocess.PIPE,
    )
    try:
      stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
      process.terminate()
      await process.wait()
      raise
    if process.returncode == 0:
      return
    details = stderr or stdout
    message = details.decode("utf-8", errors="replace").strip()[-2000:]
    raise RuntimeError(message or f"Runtime installer exited with {process.returncode}.")
