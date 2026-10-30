"""Voice Open fatal-Nari process supervisor tests."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app import supervisor as supervise_server


def test_supervisor_restarts_only_the_fatal_runtime_exit_code(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Restart a poisoned CUDA process once and return the healthy child result."""
  exit_codes = iter((supervise_server.SUPERVISOR_RESTART_EXIT_CODE, 0))
  children: list[SimpleNamespace] = []
  sleeps: list[int] = []

  def launch(command: list[str], *, env: dict[str, str]) -> SimpleNamespace:
    """Create one deterministic child-process substitute."""
    child = SimpleNamespace(
      command=command,
      environment=env,
      wait=lambda timeout=None: next(exit_codes),
    )
    children.append(child)
    return child

  monkeypatch.setattr(supervise_server.subprocess, "Popen", launch)
  monkeypatch.setattr(supervise_server.time, "sleep", sleeps.append)
  monkeypatch.setenv("VOICE_MODEL_SELECTION_FILE", str(tmp_path / "selection.json"))

  result = supervise_server.supervise(["--ui", "--host", "0.0.0.0"])

  assert result == 0
  assert len(children) == 2
  assert children[0].command[-2:] == ["--host", "0.0.0.0"]
  assert children[0].environment["VOICE_NARI_EXIT_ON_FAILURE"] == "true"
  assert sleeps == [1]


def test_supervisor_does_not_restart_an_ordinary_configuration_failure(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Preserve normal nonzero exits instead of hiding startup mistakes."""
  child = SimpleNamespace(wait=lambda timeout=None: 2)
  monkeypatch.setattr(supervise_server.subprocess, "Popen", lambda *args, **kwargs: child)
  monkeypatch.setenv("VOICE_MODEL_SELECTION_FILE", str(tmp_path / "selection.json"))

  assert supervise_server.supervise(["--ui"]) == 2


def test_supervisor_immediately_restarts_an_intentional_model_switch(
  monkeypatch: pytest.MonkeyPatch,
  tmp_path: Path,
) -> None:
  """Do not apply fatal-failure backoff to an ordinary TTS selection change."""
  exit_codes = iter((supervise_server.MODEL_SWITCH_RESTART_EXIT_CODE, 0))
  launches: list[dict[str, str]] = []

  def launch(command: list[str], *, env: dict[str, str]) -> SimpleNamespace:
    """Capture child settings and return the next deterministic exit."""
    del command
    launches.append(env)
    return SimpleNamespace(wait=lambda timeout=None: next(exit_codes))

  selection = tmp_path / "selection.json"
  monkeypatch.setenv("VOICE_MODEL_SELECTION_FILE", str(selection))
  monkeypatch.setattr(supervise_server.subprocess, "Popen", launch)

  assert supervise_server.supervise(["--ui"]) == 0
  assert len(launches) == 2
  assert launches[0]["VOICE_SUPERVISOR_ENABLED"] == "true"
  assert launches[0]["VOICE_MODEL_SELECTION_FILE"] == str(selection.resolve())
