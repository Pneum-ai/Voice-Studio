"""Configuration parsing and optional UI safety tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings


def test_empty_optional_environment_paths_remain_none(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Treat portable empty .env values as unset rather than the current directory."""
  monkeypatch.setenv("VOICE_MODEL_CACHE_DIR", "")
  monkeypatch.setenv("VOICE_DATABASE_FILE", "")
  monkeypatch.setenv("VOICE_SSL_CERTFILE", "")
  monkeypatch.setenv("VOICE_SSL_KEYFILE", "")
  settings = Settings()
  assert settings.database_file is None
  assert settings.model_cache_dir is None
  assert settings.ssl_certfile is None
  assert settings.ssl_keyfile is None


def test_explicit_database_file_is_independent_of_data_dir(tmp_path: Path) -> None:
  """Allow WSL to keep SQLite native while larger assets remain on Windows."""
  data_dir = tmp_path / "assets"
  database_file = tmp_path / "native" / "voice.db"

  settings = Settings(data_dir=data_dir, database_file=database_file)

  assert settings.database_path == database_file
  assert settings.sample_dir == data_dir / "audio" / "samples"


def test_ui_mode_requires_a_production_build(tmp_path: Path) -> None:
  """Fail clearly when --ui points at a checkout without built assets."""
  settings = Settings(ui_enabled=True, ui_dist_dir=tmp_path / "missing")
  with pytest.raises(ValueError, match="npm --prefix ui run build"):
    settings.validate_ui_assets()


def test_legacy_whisper_default_setting_is_migrated(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Preserve existing deployments while using generic configuration wording."""
  monkeypatch.delenv("VOICE_DEFAULT_TRANSCRIPTION_MODEL", raising=False)
  monkeypatch.setenv("VOICE_DEFAULT_WHISPER_MODEL", "whisper-small")
  assert Settings().default_transcription_model == "whisper-small"


def test_nari_chunk_schedule_accepts_json_environment_value(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Allow benchmarked per-card codec ramps through normal settings."""
  monkeypatch.setenv("VOICE_NARI_CHUNK_SCHEDULE", "[2,4,8,12]")

  settings = Settings()

  assert settings.nari_chunk_schedule == (2, 4, 8, 12)


def test_nari_capture_tuning_requires_auto_profile(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Reject ambiguous combinations of a named and bounded capture profile."""
  monkeypatch.setenv("VOICE_NARI_CAPTURE_MAX_BATCH", "12")
  monkeypatch.setenv("VOICE_NARI_EXECUTION_PROFILE", "ttfa")

  with pytest.raises(ValueError, match="requires the auto execution profile"):
    Settings()
