"""Enforce repository source size, width, and indentation constraints."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_SUFFIXES = {
  ".css",
  ".html",
  ".js",
  ".jsx",
  ".py",
  ".sh",
  ".ts",
  ".tsx",
}
IGNORED_PARTS = {
  ".git",
  ".mypy_cache",
  ".pytest_cache",
  ".ruff_cache",
  ".venv",
  "__pycache__",
  "build",
  "data",
  "dist",
  "node_modules",
  "vendor",
}
IGNORED_PREFIXES = (".tmp", ".venv")
MAX_LINES = 500
MAX_WIDTH = 100


def source_files() -> list[Path]:
  """Return code files while pruning large ignored trees before traversal."""
  discovered: list[Path] = []
  for directory, names, files in os.walk(ROOT):
    names[:] = sorted(
      name for name in names if name not in IGNORED_PARTS and not name.startswith(IGNORED_PREFIXES)
    )
    if Path(directory) == ROOT / "app":
      names[:] = [name for name in names if name != "ui_dist"]
    root = Path(directory)
    discovered.extend(root / name for name in files if Path(name).suffix in CODE_SUFFIXES)
  return sorted(discovered)


def violations(path: Path) -> list[str]:
  """Return every source-policy violation for one file."""
  relative = path.relative_to(ROOT)
  lines = path.read_text(encoding="utf-8").splitlines()
  problems: list[str] = []
  if len(lines) > MAX_LINES:
    problems.append(f"{relative}: {len(lines)} lines exceeds {MAX_LINES}")
  for line_number, line in enumerate(lines, start=1):
    if "\t" in line:
      problems.append(f"{relative}:{line_number}: hard tab")
    if len(line) > MAX_WIDTH:
      problems.append(f"{relative}:{line_number}: width {len(line)} exceeds {MAX_WIDTH}")
  return problems


def main() -> int:
  """Print violations and return a failing process status when any exist."""
  problems = [problem for path in source_files() for problem in violations(path)]
  if problems:
    print("\n".join(problems))
    return 1
  print(f"Checked {len(source_files())} source files.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
