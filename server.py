"""Command-line entry point for Pneum.ai Voice Studio."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import uvicorn

from app.core.config import Settings
from app.core.logging import configure_logging
from app.main import create_app


def build_parser(*, ui_default: bool = False) -> argparse.ArgumentParser:
  """Create the documented command-line parser.

  Args:
    ui_default: Serve the bundled UI unless the caller explicitly uses the API entry point.

  Returns:
    The configured argument parser.
  """
  parser = argparse.ArgumentParser(description="Run Pneum.ai Voice Studio and its local voice API.")
  parser.add_argument("--host", help="Bind host; defaults to VOICE_HOST.")
  parser.add_argument("--port", type=int, help="Bind port; defaults to 13337.")
  parser.add_argument("--data-dir", type=Path, help="Runtime data directory.")
  parser.add_argument("--model-cache-dir", type=Path, help="Hugging Face cache root.")
  parser.add_argument("--ssl-certfile", type=Path, help="Optional TLS certificate.")
  parser.add_argument("--ssl-keyfile", type=Path, help="Optional TLS private key.")
  prewarm = parser.add_mutually_exclusive_group()
  prewarm.add_argument(
    "--prewarm",
    action="store_true",
    default=None,
    help="Download and load the configured default models during startup.",
  )
  prewarm.add_argument(
    "--no-prewarm",
    action="store_false",
    dest="prewarm",
    help="Do not download or load models during startup (the default).",
  )
  parser.add_argument(
    "--shutdown-timeout",
    type=int,
    help="Seconds to drain active requests before shutdown.",
  )
  parser.add_argument(
    "--log-level",
    choices=("debug", "info", "warning", "error"),
    help="Console logging threshold; defaults to VOICE_LOG_LEVEL.",
  )
  parser.add_argument(
    "--ui",
    action="store_true",
    default=ui_default,
    help="Serve the bundled UI build.",
  )
  return parser


def settings_from_args(arguments: argparse.Namespace) -> Settings:
  """Merge explicit CLI values over environment-backed settings."""
  overrides: dict[str, Any] = {}
  for field in (
    "host",
    "port",
    "data_dir",
    "model_cache_dir",
    "ssl_certfile",
    "ssl_keyfile",
    "shutdown_timeout_seconds",
    "log_level",
  ):
    argument_name = "shutdown_timeout" if field == "shutdown_timeout_seconds" else field
    value = getattr(arguments, argument_name)
    if value is not None:
      overrides[field] = value
  if arguments.prewarm is not None:
    overrides["prewarm"] = arguments.prewarm
  if arguments.ui:
    overrides["ui_enabled"] = True
  return Settings(**overrides)


def run(arguments: argparse.Namespace) -> None:
  """Validate parsed configuration and run Uvicorn until process shutdown.

  Args:
    arguments: Parsed Voice Open command-line arguments.
  """
  settings = settings_from_args(arguments)
  configure_logging(settings.log_level)
  try:
    settings.validate_tls_pair()
    settings.validate_ui_assets()
  except ValueError as error:
    raise SystemExit(str(error)) from error
  uvicorn.run(
    create_app(settings),
    host=settings.host,
    port=settings.port,
    ssl_certfile=str(settings.ssl_certfile) if settings.ssl_certfile else None,
    ssl_keyfile=str(settings.ssl_keyfile) if settings.ssl_keyfile else None,
    timeout_graceful_shutdown=settings.shutdown_timeout_seconds,
    log_config=None,
    log_level=settings.log_level.lower(),
  )


def main() -> None:
  """Run the API-first Python entry point until process shutdown."""
  run(build_parser().parse_args())


def studio_main() -> None:
  """Run the installed Voice Open command with the bundled UI enabled."""
  run(build_parser(ui_default=True).parse_args())


if __name__ == "__main__":
  main()
