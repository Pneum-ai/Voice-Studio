"""Build release archives and reject incomplete or private package contents."""

from __future__ import annotations

import shutil
import subprocess
import tarfile
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WHEEL_REQUIRED = {
  "app/ui_dist/index.html",
  "nari_qwen3_tts/profiles/balanced.yaml",
}
LICENSE_FILES = ("LICENSE", "THIRD_PARTY_NOTICES.md", "vendor/nari-qwen3-tts/LICENSE")


def _reject_private_files(names: list[str], *, artifact: Path) -> None:
  """Raise when an archive contains local state or raw benchmark output."""
  exposed = [
    name
    for name in names
    if Path(name).name == ".env"
    or ".local." in Path(name).name
    or Path(name).name.startswith("benchmark-")
    or "runpod-benchmark-results" in Path(name).parts
  ]
  if exposed:
    raise RuntimeError(f"{artifact.name} contains private or bulky files: {exposed}")


def verify_wheel(path: Path) -> None:
  """Verify UI, Nari, commands, and exclusions in one built wheel."""
  with zipfile.ZipFile(path) as archive:
    names = archive.namelist()
    for relative in LICENSE_FILES:
      bundled = next(
        (name for name in names if name.endswith(f".dist-info/licenses/{relative}")), None
      )
      if bundled is None or archive.read(bundled) != (ROOT / relative).read_bytes():
        raise RuntimeError(f"{path.name} has missing or outdated license text: {relative}")
    metadata_name = next(name for name in names if name.endswith(".dist-info/METADATA"))
    if "License-Expression: Apache-2.0" not in archive.read(metadata_name).decode().splitlines():
      raise RuntimeError(f"{path.name} does not declare the Apache-2.0 project license.")
    missing = sorted(WHEEL_REQUIRED.difference(names))
    if missing:
      raise RuntimeError(f"{path.name} is missing required files: {missing}")
    if not any(name.startswith("app/ui_dist/assets/") for name in names):
      raise RuntimeError(f"{path.name} contains no fingerprinted UI assets.")
    development_files = [
      name for name in names if name.startswith(("docs/", "evals/", "scripts/", "tests/"))
    ]
    if development_files:
      raise RuntimeError(f"{path.name} contains source-only files: {development_files}")
    _reject_private_files(names, artifact=path)
    entry_name = next(name for name in names if name.endswith(".dist-info/entry_points.txt"))
    entries = archive.read(entry_name).decode("utf-8")
  for command in ("voice-open =", "voice-open-api ="):
    if command not in entries:
      raise RuntimeError(f"{path.name} is missing the {command[:-2]} command.")
  if "app =" in entries:
    raise RuntimeError(f"{path.name} still exposes the generic app command.")


def verify_sdist(path: Path) -> None:
  """Verify a source archive can reproduce the bundled runtime and UI."""
  with tarfile.open(path, mode="r:gz") as archive:
    names = archive.getnames()
    for relative in LICENSE_FILES:
      member = next((name for name in names if name.endswith(f"/{relative}")), None)
      contents = archive.extractfile(member) if member is not None else None
      if contents is None or contents.read() != (ROOT / relative).read_bytes():
        raise RuntimeError(f"{path.name} has missing or outdated license text: {relative}")
  _reject_private_files(names, artifact=path)
  required_suffixes = (
    "/app/ui_dist/index.html",
    "/vendor/nari-qwen3-tts/LICENSE",
    "/vendor/nari-qwen3-tts/src/nari_qwen3_tts/__init__.py",
  )
  missing = [
    suffix for suffix in required_suffixes if not any(name.endswith(suffix) for name in names)
  ]
  if missing:
    raise RuntimeError(f"{path.name} is missing required source files: {missing}")


def main() -> None:
  """Build an sdist and wheel in isolation, then inspect both archives."""
  uv = shutil.which("uv")
  if uv is None:
    raise RuntimeError("uv is required to verify release artifacts.")
  with tempfile.TemporaryDirectory(prefix="voice-open-dist-") as directory:
    output = Path(directory)
    try:
      subprocess.run(
        (uv, "build", "--offline", "--out-dir", str(output)),
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
      )
    except subprocess.CalledProcessError as error:
      details = "\n".join(value for value in (error.stdout, error.stderr) if value)
      raise RuntimeError(f"Distribution build failed:\n{details}") from error
    wheel = next(output.glob("*.whl"))
    sdist = next(output.glob("*.tar.gz"))
    verify_wheel(wheel)
    verify_sdist(sdist)
  print("Distribution archives contain the bundled UI and Nari runtime without local data.")


if __name__ == "__main__":
  main()
