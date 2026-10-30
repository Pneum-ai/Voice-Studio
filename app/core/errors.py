"""Typed service errors and RFC-style problem responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ServiceError(Exception):
  """Represent an expected service failure with a stable public error code."""

  status: int
  code: str
  title: str
  detail: str
  errors: list[dict[str, Any]] | None = None
  retry_after_ms: int | None = None


class ConflictError(ServiceError):
  """Report a resource conflict such as a duplicate profile name."""

  def __init__(self, detail: str, code: str = "conflict") -> None:
    """Initialize a conflict error."""
    super().__init__(409, code, "Conflict", detail)


class ModelNotReadyError(ServiceError):
  """Report that a selected inference model is not loaded."""

  def __init__(self, model_id: str) -> None:
    """Initialize an actionable model readiness error."""
    super().__init__(
      409,
      "model_not_ready",
      "Model is not ready",
      f"Load {model_id} before running inference.",
    )


class OverloadedError(ServiceError):
  """Reject work before an inference queue can destabilize the runtime."""

  def __init__(self, retry_after_ms: int) -> None:
    """Initialize a retryable capacity response with bounded delay guidance."""
    super().__init__(
      429,
      "overloaded",
      "Speech service is busy",
      f"Retry after approximately {retry_after_ms} ms.",
    )
    self.retry_after_ms = retry_after_ms


class InferenceTimeoutError(ServiceError):
  """Report a bounded inference deadline instead of an internal error."""

  def __init__(self, detail: str) -> None:
    """Initialize an explicit gateway-style inference timeout."""
    super().__init__(504, "inference_timeout", "Speech generation timed out", detail)


class InferenceUnavailableError(ServiceError):
  """Report a loaded runtime whose execution engine cannot accept work."""

  def __init__(self, reason: str, failure_id: str | None = None) -> None:
    """Initialize a correlated runtime-readiness failure."""
    suffix = f" Failure ID: {failure_id}." if failure_id else ""
    super().__init__(
      503,
      "inference_unavailable",
      "Speech inference is unavailable",
      f"The speech runtime is not ready: {reason}.{suffix}",
    )


class NotFoundError(ServiceError):
  """Report that a requested resource does not exist."""

  def __init__(self, resource: str) -> None:
    """Initialize a resource-not-found error."""
    super().__init__(404, "not_found", "Resource not found", f"{resource} was not found.")


class ValidationError(ServiceError):
  """Report a domain validation failure."""

  def __init__(self, detail: str, code: str = "invalid_request") -> None:
    """Initialize a validation error."""
    super().__init__(400, code, "Invalid request", detail)
