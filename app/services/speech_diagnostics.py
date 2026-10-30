"""Request-local collection of optional streaming synthesis diagnostics."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

DiagnosticValue = float | bool
Diagnostics = dict[str, DiagnosticValue]
_CURRENT: ContextVar[Diagnostics | None] = ContextVar("speech_diagnostics", default=None)


@contextmanager
def collect_diagnostics(enabled: bool) -> Iterator[Diagnostics | None]:
  """Collect backend timings in the current asynchronous request context."""
  value: Diagnostics | None = {} if enabled else None
  token = _CURRENT.set(value)
  try:
    yield value
  finally:
    _CURRENT.reset(token)


def record_diagnostics(values: Diagnostics) -> None:
  """Merge backend measurements when the caller opted into diagnostics."""
  current = _CURRENT.get()
  if current is not None:
    current.update(values)


__all__ = ["collect_diagnostics", "record_diagnostics"]
