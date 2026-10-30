"""Blackwell FP8 selection behavior."""

from __future__ import annotations

from types import SimpleNamespace

from nari_qwen3_tts.executor import optimizations


class _FakeMLP:
  """Expose the minimal projection optimization surface used by the selector."""

  def __init__(self) -> None:
    """Start with BF16 selected and no quantized storage."""
    self.fp8_backend: str | None = None
    self.disabled = False

  def initialize_fp8_gate_up_weight(self, *, blackwell: bool = False) -> bool:
    """Pretend the requested architecture has a supported kernel."""
    assert blackwell is True
    self.fp8_backend = "sm120_scaled_mm"
    return True

  def disable_fp8_gate_up_weight(self) -> None:
    """Record automatic rejection and restore BF16 selection."""
    self.disabled = True
    self.fp8_backend = None


def _layers(count: int) -> tuple[SimpleNamespace, ...]:
  """Build fake transformer layers with projection-compatible MLPs."""
  return tuple(SimpleNamespace(mlp=_FakeMLP()) for _ in range(count))


def test_blackwell_auto_rejects_a_slower_projection(monkeypatch) -> None:
  """Keep BF16 when capture-safe FP8 loses the batch-one microbenchmark."""
  layers = _layers(2)
  monkeypatch.setattr(optimizations, "_benchmark_blackwell", lambda _: (0.2, 0.01))

  enabled, speedup, relative_error = optimizations._install_blackwell_fp8(
    layers,
    mode="auto",
  )

  assert enabled == 0
  assert speedup == 0.2
  assert relative_error == 0.01
  assert layers[0].mlp.disabled is True
  assert layers[1].mlp.fp8_backend is None


def test_blackwell_auto_falls_back_when_probe_fails(monkeypatch) -> None:
  """Keep model startup operational when the local scaled GEMM probe errors."""
  layers = _layers(2)

  def fail_probe(_: object) -> tuple[float, float]:
    """Simulate an unsupported or unhealthy local FP8 implementation."""
    raise RuntimeError("scaled GEMM unavailable")

  monkeypatch.setattr(optimizations, "_benchmark_blackwell", fail_probe)

  enabled, speedup, relative_error = optimizations._install_blackwell_fp8(
    layers,
    mode="auto",
  )

  assert enabled == 0
  assert speedup is None
  assert relative_error is None
  assert layers[0].mlp.disabled is True
  assert layers[1].mlp.fp8_backend is None


def test_blackwell_force_enables_all_layers(monkeypatch) -> None:
  """Keep an explicit FP8 override available on compatible cards."""
  layers = _layers(3)
  monkeypatch.setattr(optimizations, "_benchmark_blackwell", lambda _: (0.2, 0.04))

  enabled, _, _ = optimizations._install_blackwell_fp8(layers, mode="force")
  assert enabled == 3
