"""Run every network-free repository verification gate in CI order."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = ROOT / "ui"


def _checks() -> tuple[tuple[Path, tuple[str, ...]], ...]:
  """Return backend and frontend verification commands with working paths."""
  npm = shutil.which("npm")
  if npm is None:
    raise RuntimeError("npm is required to verify and build the bundled UI.")
  return (
    (ROOT, ("ruff", "format", "--check", ".")),
    (ROOT, ("ruff", "check", ".")),
    (ROOT, ("pyright",)),
    (ROOT, (sys.executable, "-m", "scripts.check_source")),
    (UI_ROOT, (npm, "run", "lint")),
    (UI_ROOT, (npm, "run", "test")),
    (UI_ROOT, (npm, "run", "build")),
    (ROOT, (sys.executable, "-m", "pytest")),
    (ROOT, (sys.executable, "-m", "scripts.verify_distribution")),
  )


def main() -> None:
  """Run all gates and stop immediately when one fails."""
  for working_directory, command in _checks():
    print(f"> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=working_directory, check=True)


if __name__ == "__main__":
  main()
