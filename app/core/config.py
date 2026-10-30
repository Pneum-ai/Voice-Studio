"""Application configuration and runtime path management."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGED_UI_DIR = Path(__file__).resolve().parents[1] / "ui_dist"


class Settings(BaseSettings):
  """Validated service settings loaded from environment variables."""

  host: str = "127.0.0.1"
  port: int = Field(default=13337, ge=1, le=65535)
  data_dir: Path = Path("data")
  database_file: Path | None = None
  model_cache_dir: Path | None = None
  default_qwen_model: str = "qwen-tts-0.6B"
  default_transcription_model: str = "whisper-turbo"
  default_whisper_model: str | None = Field(default=None, exclude=True)
  qwen_asr_python: Path | None = None
  auto_install_runtimes: bool = True
  torch_compile: bool = False
  qwen_engine: Literal["auto", "nari", "official"] = "nari"
  nari_blackwell_fp8: Literal["off", "auto", "force"] = "auto"
  nari_capture_max_batch: int | None = Field(default=None, ge=1, le=32)
  nari_execution_profile: Literal["auto", "rtx_24gb", "ttfa"] = "auto"
  nari_policy_mode: Literal["safe", "candidate"] = "safe"
  nari_max_active_0_6b: int = Field(default=8, ge=1, le=128)
  nari_max_active_1_7b: int = Field(default=4, ge=1, le=128)
  nari_max_queued: int = Field(default=8, ge=0, le=128)
  nari_queue_timeout_ms: int = Field(default=5_000, ge=1, le=10_000)
  nari_first_audio_timeout_seconds: float = Field(default=3.0, gt=0, le=60)
  nari_request_timeout_seconds: float = Field(default=60.0, gt=0, le=600)
  nari_silence_trim: bool = True
  nari_silence_threshold_dbfs: float = Field(default=-45.0, ge=-90, le=-10)
  nari_silence_max_ms: int = Field(default=3_000, ge=100, le=10_000)
  nari_silence_preroll_ms: int = Field(default=10, ge=0, le=100)
  nari_silence_min_output_ms: int = Field(default=160, ge=0, le=1_000)
  nari_chunk_schedule: tuple[int, ...] = (4, 4, 8, 12)
  nari_low_latency_chunk_schedule: tuple[int, ...] = (1, 2, 4, 8, 12)
  nari_low_latency_max_active: int = Field(default=1, ge=0, le=128)
  nari_adaptive_target_ms: int = Field(default=500, ge=50, le=10_000)
  nari_watchdog_stall_seconds: float = Field(default=90.0, gt=1, le=300)
  nari_exit_on_failure: bool = False
  supervisor_enabled: bool = Field(default=False, exclude=True)
  model_selection_file: Path | None = Field(default=None, exclude=True)
  stream_send_timeout_seconds: float = Field(default=5.0, gt=0, le=60)
  batch_enabled: bool = True
  batch_max_size: int = Field(default=8, ge=1, le=32)
  batch_window_ms: int = Field(default=50, ge=0, le=1000)
  prewarm: bool = False
  ssl_certfile: Path | None = None
  ssl_keyfile: Path | None = None
  cors_origins: tuple[str, ...] = ()
  cors_origin_regex: str | None = r"^https?://(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?$"
  ui_enabled: bool = False
  ui_dist_dir: Path = PACKAGED_UI_DIR
  max_upload_bytes: int = 100 * 1024 * 1024
  job_retention_seconds: int = 300
  shutdown_timeout_seconds: int = Field(default=10, ge=1, le=120)
  log_level: str = "INFO"

  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    env_prefix="VOICE_",
    env_ignore_empty=True,
    extra="ignore",
  )

  @model_validator(mode="before")
  @classmethod
  def migrate_default_whisper_model(cls, value: object) -> object:
    """Accept the deprecated Whisper-specific setting when the new key is absent."""
    if not isinstance(value, dict):
      return value
    migrated = dict(value)
    if "default_transcription_model" not in migrated and migrated.get("default_whisper_model"):
      migrated["default_transcription_model"] = migrated["default_whisper_model"]
    return migrated

  @field_validator("cors_origins", mode="before")
  @classmethod
  def parse_cors_origins(cls, value: object) -> object:
    """Parse a comma-separated CORS origin environment value."""
    if isinstance(value, str):
      return tuple(item.strip() for item in value.split(",") if item.strip())
    return value

  @field_validator("nari_chunk_schedule", "nari_low_latency_chunk_schedule")
  @classmethod
  def validate_nari_chunk_schedule(cls, value: tuple[int, ...]) -> tuple[int, ...]:
    """Require a nonempty codec ramp supported by captured Nari frame sizes."""
    if not value or any(item < 1 or item > 12 for item in value):
      raise ValueError("Nari chunk schedule entries must be between 1 and 12.")
    return value

  @model_validator(mode="after")
  def validate_nari_capture_override(self) -> Settings:
    """Require automatic profile resolution for a bounded tuning capture."""
    if self.nari_capture_max_batch is not None and self.nari_execution_profile != "auto":
      raise ValueError("Nari capture batch tuning requires the auto execution profile.")
    return self

  @field_validator("log_level", mode="before")
  @classmethod
  def validate_log_level(cls, value: object) -> str:
    """Normalize and validate the console logging threshold."""
    normalized = str(value).upper()
    if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
      raise ValueError("Log level must be DEBUG, INFO, WARNING, or ERROR.")
    return normalized

  @property
  def database_path(self) -> Path:
    """Return the explicit database file or the data-directory default."""
    return self.database_file or self.data_dir / "voice.db"

  @property
  def sample_dir(self) -> Path:
    """Return the profile sample audio directory."""
    return self.data_dir / "audio" / "samples"

  @property
  def generation_dir(self) -> Path:
    """Return the saved generation audio directory."""
    return self.data_dir / "audio" / "generations"

  @property
  def prompt_cache_dir(self) -> Path:
    """Return the prompt cache directory."""
    return self.data_dir / "cache" / "prompts"

  @property
  def reference_cache_dir(self) -> Path:
    """Return the combined reference cache directory."""
    return self.data_dir / "cache" / "references"

  @property
  def runtime_dir(self) -> Path:
    """Return the managed optional model-runtime directory."""
    return self.data_dir / "runtimes"

  @property
  def model_selection_path(self) -> Path:
    """Return the supervisor coordination file used for durable TTS selection."""
    return self.model_selection_file or self.data_dir / "model-selection.json"

  def create_runtime_directories(self) -> None:
    """Create all directories required for safe runtime file writes."""
    for path in (
      self.data_dir,
      self.sample_dir,
      self.generation_dir,
      self.prompt_cache_dir,
      self.reference_cache_dir,
      self.runtime_dir,
    ):
      path.mkdir(parents=True, exist_ok=True)

  def validate_tls_pair(self) -> None:
    """Raise when only one member of the TLS certificate pair is configured."""
    if bool(self.ssl_certfile) != bool(self.ssl_keyfile):
      raise ValueError("TLS certificate and key must be configured together.")

  def validate_ui_assets(self) -> None:
    """Raise when UI mode is enabled without a production frontend build."""
    if self.ui_enabled and not (self.ui_dist_dir / "index.html").is_file():
      raise ValueError(
        f"UI assets not found in {self.ui_dist_dir}. Run `npm --prefix ui run build` "
        "before using --ui."
      )
