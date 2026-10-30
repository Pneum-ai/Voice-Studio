"""Command-line startup setting behavior."""

from server import build_parser, settings_from_args


def test_prewarm_is_disabled_by_default() -> None:
  """Avoid downloading model weights during a first launch."""
  arguments = build_parser().parse_args([])

  assert settings_from_args(arguments).prewarm is False


def test_installed_studio_command_enables_ui_by_default() -> None:
  """Serve the complete studio when launched through the project command."""
  arguments = build_parser(ui_default=True).parse_args([])

  assert settings_from_args(arguments).ui_enabled is True


def test_prewarm_enables_startup_model_loading() -> None:
  """Map the explicit opt-in flag to startup model loading."""
  arguments = build_parser().parse_args(["--prewarm"])

  assert settings_from_args(arguments).prewarm is True


def test_shutdown_timeout_is_configurable() -> None:
  """Pass the graceful request-drain timeout into runtime settings."""
  arguments = build_parser().parse_args(["--shutdown-timeout", "7"])

  assert settings_from_args(arguments).shutdown_timeout_seconds == 7


def test_log_level_is_normalized_from_cli() -> None:
  """Apply the selected logcat-style console threshold."""
  arguments = build_parser().parse_args(["--log-level", "debug"])

  assert settings_from_args(arguments).log_level == "DEBUG"
