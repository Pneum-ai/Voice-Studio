"""Card-specific Nari capture fallback and serving-policy tests."""

from __future__ import annotations

from dataclasses import replace

import pytest

from app.backends.pytorch.nari_policy import (
  build_with_capture_fallback,
  resolve_nari_policy,
)
from app.backends.pytorch.nari_streaming import NariRuntimeConfig


@pytest.mark.parametrize("model_id", ["qwen-tts-1.7B", "qwen-tts-1.7B-customvoice"])
def test_4090_1_7b_starts_below_failed_code_predictor_batch(model_id: str) -> None:
  """Avoid the observed batch-five OOM and retain a smaller fallback."""
  plan = resolve_nari_policy(
    NariRuntimeConfig(),
    device_name="NVIDIA GeForce RTX 4090",
    memory_in_gb=24,
    model_id=model_id,
  )

  primary = plan.candidates[0]
  fallback = plan.candidates[1]
  primary_batch = primary.profile.stages.code_predictor.max_batch_size
  assert 0 < fallback.profile.stages.code_predictor.max_batch_size < primary_batch <= 4
  assert primary.profile.stages.talker_prefill.max_batch_size <= 4


def test_tuning_capture_override_is_single_and_exact() -> None:
  """Benchmark one requested geometry without silently using a fallback."""
  config = replace(NariRuntimeConfig(), capture_max_batch=12)
  plan = resolve_nari_policy(
    config,
    device_name="NVIDIA GeForce RTX 5090",
    memory_in_gb=32,
    model_id="qwen-tts-0.6B",
  )

  assert plan.name == "tuned-b12"
  assert len(plan.candidates) == 1
  assert plan.candidates[0].profile.stages.code_predictor.max_batch_size == 12


def test_capture_failure_uses_next_candidate_and_records_fallback() -> None:
  """Recover a contextual CUDA capture failure without hiding selected identity."""
  plan = resolve_nari_policy(
    NariRuntimeConfig(),
    device_name="NVIDIA GeForce RTX 4090",
    memory_in_gb=24,
    model_id="qwen-tts-1.7B",
  )
  attempts: list[str] = []
  recoveries = 0

  def build(candidate: object) -> object:
    """Fail the observed large profile and accept its bounded fallback."""
    name = getattr(candidate, "name")
    attempts.append(name)
    if len(attempts) == 1:
      raise RuntimeError("CUDA capture failed while building bounded graph")
    return object()

  def recover() -> None:
    """Count cleanup between graph capture attempts."""
    nonlocal recoveries
    recoveries += 1

  result = build_with_capture_fallback(plan, build, recover)

  assert attempts == [candidate.name for candidate in plan.candidates[:2]]
  assert recoveries == 1
  assert result.capture_profile == plan.candidates[1].name
  assert result.capture_fallbacks == 1


def test_non_capture_failure_is_not_retried() -> None:
  """Avoid masking checkpoint, adapter, or request-contract failures."""
  plan = resolve_nari_policy(
    NariRuntimeConfig(),
    device_name="NVIDIA H100 80GB HBM3",
    memory_in_gb=80,
    model_id="qwen-tts-1.7B",
  )

  with pytest.raises(ValueError, match="adapter metadata"):
    build_with_capture_fallback(
      plan,
      lambda candidate: (_ for _ in ()).throw(ValueError("adapter metadata invalid")),
      lambda: None,
    )
