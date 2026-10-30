"""Prepare Linux CUDA linker paths required by the Nari runtime."""

from __future__ import annotations

import os
import sys
import sysconfig
from pathlib import Path


def _cuda_link_directory(toolkit: Path) -> Path | None:
  """Create unversioned linker aliases omitted by the pip CUDA packages."""
  runtimes = sorted((toolkit / "lib").glob("libcudart.so.*"))
  if not runtimes:
    return None
  link_directory = Path.home() / ".cache" / "voice-open" / "cuda-links"
  link_directory.mkdir(parents=True, exist_ok=True)
  aliases = {
    "libcudart.so": runtimes[0],
    "libcuda.so": Path("/usr/lib/wsl/lib/libcuda.so"),
  }
  for name, target in aliases.items():
    link = link_directory / name
    if not target.is_file():
      continue
    if link.is_symlink():
      if link.readlink() == target:
        continue
      link.unlink()
    if not link.exists():
      link.symlink_to(target)
  return link_directory


def configure_cuda_toolkit() -> None:
  """Expose the uv-installed CUDA compiler to FlashInfer's JIT loader."""
  configured_home = os.environ.get("CUDA_HOME")
  if configured_home:
    toolkit = Path(configured_home)
  else:
    purelib = Path(sysconfig.get_paths()["purelib"])
    candidates = sorted((purelib / "nvidia").glob("cu*/bin/nvcc"), reverse=True)
    if not candidates:
      return
    toolkit = candidates[0].parents[1]
    os.environ["CUDA_HOME"] = str(toolkit)
  executable_bin = Path(sys.executable).parent
  os.environ["PATH"] = os.pathsep.join(
    (str(toolkit / "bin"), str(executable_bin), os.environ.get("PATH", ""))
  )
  link_directory = _cuda_link_directory(toolkit)
  library_paths = [link_directory] if link_directory is not None else []
  library_paths.extend(path for path in (toolkit / "lib", toolkit / "lib64") if path.is_dir())
  wsl_driver_path = Path("/usr/lib/wsl/lib")
  if wsl_driver_path.is_dir():
    library_paths.append(wsl_driver_path)
  for variable in ("LIBRARY_PATH", "LD_LIBRARY_PATH"):
    existing = os.environ.get(variable, "")
    os.environ[variable] = os.pathsep.join(
      [str(path) for path in library_paths] + ([existing] if existing else [])
    )


__all__ = ["configure_cuda_toolkit"]
