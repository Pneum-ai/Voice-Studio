"""Restart Voice Open after a fatal Nari CUDA worker failure."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from app.backends.pytorch.nari_streaming import SUPERVISOR_RESTART_EXIT_CODE
from app.services.model_selection import (
  MODEL_SWITCH_RESTART_EXIT_CODE,
  has_pending_model_switch,
)

MAX_RESTARTS = 3
RESTART_WINDOW_SECONDS = 600


def _child_command(arguments: list[str]) -> list[str]:
  """Build the child server command without another supervisor layer."""
  server = Path(__file__).parents[1] / "server.py"
  return [sys.executable, str(server), *arguments]


def _model_selection_path(arguments: list[str], environment: dict[str, str]) -> Path:
  """Resolve the coordination file consistently with child settings."""
  configured = environment.get("VOICE_MODEL_SELECTION_FILE")
  if configured:
    return Path(configured).resolve()
  data_dir = environment.get("VOICE_DATA_DIR", "data")
  for index, argument in enumerate(arguments):
    if argument == "--data-dir" and index + 1 < len(arguments):
      data_dir = arguments[index + 1]
    elif argument.startswith("--data-dir="):
      data_dir = argument.partition("=")[2]
  return (Path(data_dir) / "model-selection.json").resolve()


def supervise(arguments: list[str]) -> int:
  """Run the server and restart only explicit fatal-runtime or model-switch exits."""
  restart_times: list[float] = []
  parent_environment = dict(os.environ)
  selection_path = _model_selection_path(arguments, parent_environment)
  while True:
    environment = dict(parent_environment)
    environment["VOICE_NARI_EXIT_ON_FAILURE"] = "true"
    environment["VOICE_SUPERVISOR_ENABLED"] = "true"
    environment["VOICE_MODEL_SELECTION_FILE"] = str(selection_path)
    child = subprocess.Popen(_child_command(arguments), env=environment)
    try:
      exit_code = child.wait()
    except KeyboardInterrupt:
      child.terminate()
      try:
        child.wait(timeout=10)
      except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
      return 130
    model_switch = exit_code == MODEL_SWITCH_RESTART_EXIT_CODE
    pending_switch = has_pending_model_switch(selection_path)
    if (
      exit_code != SUPERVISOR_RESTART_EXIT_CODE
      and not model_switch
      and not (exit_code != 0 and pending_switch)
    ):
      return exit_code
    if model_switch:
      print(
        "Restarting Voice Open with a clean CUDA context for the selected TTS model.",
        file=sys.stderr,
        flush=True,
      )
      continue
    now = time.monotonic()
    restart_times = [value for value in restart_times if now - value < RESTART_WINDOW_SECONDS]
    restart_times.append(now)
    if len(restart_times) > MAX_RESTARTS:
      print(
        "Voice Open stopped after repeated CUDA worker failures; inspect the traceback.",
        file=sys.stderr,
        flush=True,
      )
      return exit_code
    delay = 2 ** (len(restart_times) - 1)
    print(
      f"Restarting Voice Open after CUDA worker failure in {delay} second(s) "
      f"({len(restart_times)}/{MAX_RESTARTS}).",
      file=sys.stderr,
      flush=True,
    )
    time.sleep(delay)


def main() -> None:
  """Supervise an API-first server invocation."""
  parser = argparse.ArgumentParser(add_help=False)
  _, remaining = parser.parse_known_args()
  raise SystemExit(supervise(remaining))


def studio_main() -> None:
  """Supervise the installed UI-enabled server invocation."""
  raise SystemExit(supervise(["--ui", *sys.argv[1:]]))


if __name__ == "__main__":
  main()
