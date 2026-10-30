"""Compact logcat-style console logging for the complete application."""

from __future__ import annotations

import logging
import os
import sys
import time
import warnings
from typing import TextIO

LEVEL_CODES = {
  logging.DEBUG: "D",
  logging.INFO: "I",
  logging.WARNING: "W",
  logging.ERROR: "E",
  logging.CRITICAL: "F",
}
LEVEL_COLORS = {
  logging.DEBUG: "\x1b[36m",
  logging.INFO: "\x1b[32m",
  logging.WARNING: "\x1b[33m",
  logging.ERROR: "\x1b[31m",
  logging.CRITICAL: "\x1b[35;1m",
}
LOGGER_TAGS = {
  "app.backends.device_info": "device",
  "app.services.container": "lifecycle",
  "app.services.models": "models",
  "py.warnings": "warning",
  "uvicorn.access": "http",
  "uvicorn.error": "server",
}
QUIET_LOGGERS = {
  "torch.utils.flop_counter": logging.ERROR,
}
RESET_COLOR = "\x1b[0m"
TAG_WIDTH = 12


def log_tag(logger_name: str) -> str:
  """Return a short stable tag for one standard-library logger name."""
  if logger_name in LOGGER_TAGS:
    return LOGGER_TAGS[logger_name]
  if logger_name.startswith("qwen_tts."):
    return "qwen"
  if logger_name.startswith("transformers."):
    return "whisper"
  return logger_name.removeprefix("app.").rsplit(".", maxsplit=1)[-1][:TAG_WIDTH]


class LogcatFormatter(logging.Formatter):
  """Render standard log records as compact Android-style console lines."""

  def __init__(self, *, color: bool) -> None:
    """Configure optional terminal colors without changing message content."""
    super().__init__()
    self.color = color

  def format(self, record: logging.LogRecord) -> str:
    """Format one record with timestamp, severity code, tag, and indentation."""
    timestamp = time.strftime("%H:%M:%S", self.converter(record.created))
    timestamp = f"{timestamp}.{int(record.msecs):03d}"
    level = LEVEL_CODES.get(record.levelno, "?")
    tag = log_tag(record.name)
    prefix = f"{timestamp} {level}/{tag:<{TAG_WIDTH}} | "
    message = record.getMessage()
    if record.exc_info:
      exception = self.formatException(record.exc_info)
      message = f"{message}\n{exception}" if message else exception
    if record.stack_info:
      message = f"{message}\n{self.formatStack(record.stack_info)}"
    indent = " " * len(prefix)
    rendered = prefix + message.replace("\n", f"\n{indent}")
    if not self.color:
      return rendered
    color = LEVEL_COLORS.get(record.levelno, "")
    return f"\x1b[2m{timestamp}\x1b[0m {color}{level}/{tag:<{TAG_WIDTH}}{RESET_COLOR} | " + (
      message.replace("\n", f"\n{indent}")
    )


def _supports_color(stream: TextIO) -> bool:
  """Return whether ANSI colors should be emitted to the configured stream."""
  return bool(
    hasattr(stream, "isatty")
    and stream.isatty()
    and not os.environ.get("NO_COLOR")
    and os.environ.get("TERM") != "dumb"
  )


def _configure_warning_filters() -> None:
  """Hide narrowly identified upstream compatibility noise."""
  warnings.filterwarnings(
    "ignore",
    message=r"`torch\.jit\.script_method` is not supported in Python 3\.14\+.*",
    category=DeprecationWarning,
    module=r"torch\.jit\._script",
  )
  warnings.filterwarnings(
    "ignore",
    message=r"aifc was removed in Python 3\.13\..*",
    category=DeprecationWarning,
    module=r"audioread\.rawread",
  )
  warnings.filterwarnings(
    "ignore",
    message=r"sunau was removed in Python 3\.13\..*",
    category=DeprecationWarning,
    module=r"audioread\.rawread",
  )


def configure_logging(level: str, stream: TextIO | None = None) -> None:
  """Route application, Uvicorn, warnings, and upstream logs through one sink."""
  output = stream or sys.stderr
  handler = logging.StreamHandler(output)
  handler.setFormatter(LogcatFormatter(color=_supports_color(output)))
  logging.basicConfig(
    level=getattr(logging, level),
    handlers=[handler],
    force=True,
  )
  logging.captureWarnings(True)
  warnings.simplefilter("default")
  _configure_warning_filters()
  for logger_name in (
    "uvicorn",
    "uvicorn.error",
    "uvicorn.access",
  ):
    target = logging.getLogger(logger_name)
    target.handlers.clear()
    target.propagate = True
    target.setLevel(getattr(logging, level))
  for logger_name in (
    "huggingface_hub",
    "httpcore",
    "httpx",
    "qwen_tts.core.models.configuration_qwen3_tts",
    "qwen_tts.core.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2",
  ):
    logging.getLogger(logger_name).setLevel(logging.WARNING)
  for logger_name, logger_level in QUIET_LOGGERS.items():
    logging.getLogger(logger_name).setLevel(logger_level)
