"""Resolve isolated Nari capture and serving policies for NVIDIA GPU classes."""

from __future__ import annotations

import gc
import logging
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol

from app.backends.pytorch.nari_streaming import NariRuntimeConfig

if TYPE_CHECKING:
  from nari_qwen3_tts.profile import ResolvedProfile

LOGGER = logging.getLogger(__name__)

_CAPTURE_FAILURE_MARKERS = (
  "capture",
  "cublas_status_not_supported",
  "cuda error",
  "cuda out of memory",
  "out of memory",
)
_FULL_BATCH_SIZES = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 16, 24, 32)


class NariGPUClass(StrEnum):
  """Identify GPU families with independently promoted Nari policies."""

  H100 = "h100"
  RTX_3090 = "rtx_3090"
  RTX_4090 = "rtx_4090"
  RTX_5090 = "rtx_5090"
  RTX_PRO_6000 = "rtx_pro_6000"
  UNKNOWN_LARGE = "unknown_large"


@dataclass(frozen=True, slots=True)
class NariCaptureCandidate:
  """Pair one auditable CUDA capture profile with its safe serving configuration."""

  name: str
  profile: ResolvedProfile
  runtime_config: NariRuntimeConfig


@dataclass(frozen=True, slots=True)
class NariPolicyPlan:
  """Describe ordered capture fallbacks for one GPU and deployment mode."""

  name: str
  gpu_class: NariGPUClass
  device_name: str
  memory_in_gb: int
  candidates: tuple[NariCaptureCandidate, ...]


@dataclass(frozen=True, slots=True)
class NariEngineBuild:
  """Return a captured engine and the exact policy that successfully built it."""

  engine: Any
  runtime_config: NariRuntimeConfig
  policy_name: str
  capture_profile: str
  capture_profile_sha256: str
  capture_fallbacks: int


class NariEngineBuilder(Protocol):
  """Build one engine attempt from a fully resolved capture candidate."""

  def __call__(self, candidate: NariCaptureCandidate) -> Any:
    """Return a captured engine or raise a contextual startup failure."""


class CaptureRecovery(Protocol):
  """Recover reclaimable CUDA allocations between capture attempts."""

  def __call__(self) -> None:
    """Release failed-attempt resources without hiding the original error."""


def classify_nari_gpu(device_name: str, memory_in_gb: int) -> NariGPUClass:
  """Classify a device using stable model names with a VRAM-safe fallback."""
  normalized = device_name.casefold()
  if "h100" in normalized:
    return NariGPUClass.H100
  if "rtx pro 6000" in normalized and "blackwell" in normalized:
    return NariGPUClass.RTX_PRO_6000
  if "rtx 5090" in normalized:
    return NariGPUClass.RTX_5090
  if "rtx 4090" in normalized:
    return NariGPUClass.RTX_4090
  if "rtx 3090" in normalized or memory_in_gb <= 26:
    return NariGPUClass.RTX_3090
  return NariGPUClass.UNKNOWN_LARGE


def _bounded_ttfa_profile(
  maximum_batch: int,
  *,
  capture_slots: int,
  kv_pages: int,
  workspace_bytes: int,
) -> ResolvedProfile:
  """Create a strict TTFA overlay that removes only unsupported large batches."""
  from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader

  batches = tuple(size for size in _FULL_BATCH_SIZES if size <= maximum_batch)
  prefill_batches = tuple(size for size in (1, 2, 4, 8) if size <= maximum_batch)
  exact_prefill_batches = tuple(range(1, min(maximum_batch, 8) + 1))
  first_frame = tuple(size for size in (1, 2, 4, 8, 16, 32) if size <= maximum_batch)
  bounded_powers = tuple(size for size in (1, 2, 4, 8) if size <= maximum_batch)
  warm_partial = tuple(size for size in range(1, min(maximum_batch, 8) + 1))
  warm_full = tuple(size for size in _FULL_BATCH_SIZES if size <= maximum_batch)
  return ProfileLoader().load_mapping(
    {
      "talker_prefill": {
        "max_batch_size": min(maximum_batch, 8),
        "batch_sizes": prefill_batches,
        "exact_batch_sizes": exact_prefill_batches,
      },
      "talker_decode": {"max_batch_size": maximum_batch, "batch_sizes": batches},
      "code_predictor": {"max_batch_size": maximum_batch, "batch_sizes": batches},
      "codec": {
        "max_batch_size": maximum_batch,
        "batches": {
          "whole_sequence_first_frame": first_frame,
          "whole_sequence_followup": bounded_powers,
          "cold": bounded_powers,
          "warm_partial": warm_partial,
          "warm_full": warm_full,
        },
      },
      "talker_capture_slots": capture_slots,
      "kv_pages": kv_pages,
      "workspace_bytes": workspace_bytes,
    },
    base=ExecutionProfile.TTFA,
  )


def _packaged_candidate(
  name: str,
  profile_name: str,
  config: NariRuntimeConfig,
) -> NariCaptureCandidate:
  """Load one immutable packaged profile and attach its serving policy."""
  from nari_qwen3_tts.profile import ExecutionProfile, ProfileLoader

  profile = ProfileLoader().load_profile(ExecutionProfile(profile_name))
  return NariCaptureCandidate(name=name, profile=profile, runtime_config=config)


def _bounded_candidate(
  name: str,
  maximum_batch: int,
  config: NariRuntimeConfig,
  *,
  capture_slots: int,
  kv_pages: int,
  workspace_bytes: int,
) -> NariCaptureCandidate:
  """Build one bounded TTFA candidate for an ordered startup fallback ladder."""
  return NariCaptureCandidate(
    name=name,
    profile=_bounded_ttfa_profile(
      maximum_batch,
      capture_slots=capture_slots,
      kv_pages=kv_pages,
      workspace_bytes=workspace_bytes,
    ),
    runtime_config=config,
  )


def _optimized_config(
  config: NariRuntimeConfig,
  gpu_class: NariGPUClass,
  model_id: str | None,
) -> NariRuntimeConfig:
  """Return opt-in serving candidates derived from completed card benchmarks."""
  if config.policy_mode != "candidate":
    return config
  if gpu_class is NariGPUClass.H100:
    return replace(
      config,
      max_active_0_6b=12,
      max_active_1_7b=8,
      max_queued=16,
      silence_min_output_ms=80,
      chunk_schedule=(1, 2, 4, 8, 12),
      adaptive_target_ms=750,
    )
  if gpu_class is NariGPUClass.RTX_PRO_6000:
    return replace(
      config,
      max_active_0_6b=10,
      max_active_1_7b=6,
      silence_min_output_ms=80,
      chunk_schedule=(2, 4, 8, 12),
      adaptive_target_ms=750,
    )
  if gpu_class is NariGPUClass.RTX_5090:
    return replace(
      config,
      silence_min_output_ms=80,
      chunk_schedule=(2, 4, 8, 12),
      adaptive_target_ms=750,
    )
  if gpu_class is NariGPUClass.RTX_4090 and model_id == "qwen-tts-0.6B":
    return replace(
      config,
      max_active_0_6b=10,
      silence_min_output_ms=80,
      chunk_schedule=(2, 4, 8, 12),
      adaptive_target_ms=750,
    )
  return config


def resolve_nari_policy(
  config: NariRuntimeConfig,
  *,
  device_name: str,
  memory_in_gb: int,
  model_id: str | None = None,
) -> NariPolicyPlan:
  """Resolve explicit overrides or a card-specific capture fallback ladder."""
  if config.capture_max_batch is not None:
    maximum_batch = config.capture_max_batch
    candidate = _bounded_candidate(
      f"tuned-b{maximum_batch}",
      maximum_batch,
      config,
      capture_slots=1 if maximum_batch <= 8 else 2 if maximum_batch <= 16 else 4,
      kv_pages=96 if maximum_batch <= 8 else 128 if maximum_batch <= 16 else 256,
      workspace_bytes=(128 if maximum_batch <= 8 else 256 if maximum_batch <= 16 else 512)
      * 1024
      * 1024,
    )
    return NariPolicyPlan(
      name=f"tuned-b{maximum_batch}",
      gpu_class=classify_nari_gpu(device_name, memory_in_gb),
      device_name=device_name,
      memory_in_gb=memory_in_gb,
      candidates=(candidate,),
    )
  if config.execution_profile != "auto":
    candidate = _packaged_candidate(
      f"explicit-{config.execution_profile}",
      config.execution_profile,
      config,
    )
    return NariPolicyPlan(
      name=f"explicit-{config.execution_profile}",
      gpu_class=classify_nari_gpu(device_name, memory_in_gb),
      device_name=device_name,
      memory_in_gb=memory_in_gb,
      candidates=(candidate,),
    )

  gpu_class = classify_nari_gpu(device_name, memory_in_gb)
  optimized = _optimized_config(config, gpu_class, model_id)
  safe_fallback = replace(
    config,
    max_active_0_6b=min(config.max_active_0_6b, 8),
    max_active_1_7b=min(config.max_active_1_7b, 4),
    max_queued=min(config.max_queued, 8),
    silence_min_output_ms=max(config.silence_min_output_ms, 160),
    chunk_schedule=(4, 4, 8, 12),
  )
  if gpu_class is NariGPUClass.RTX_4090 and model_id in {
    "qwen-tts-1.7B",
    "qwen-tts-1.7B-customvoice",
  }:
    candidates = (
      _bounded_candidate(
        "rtx_4090-1.7b-b4",
        4,
        safe_fallback,
        capture_slots=1,
        kv_pages=96,
        workspace_bytes=128 * 1024 * 1024,
      ),
      _bounded_candidate(
        "rtx_4090-1.7b-b2",
        2,
        safe_fallback,
        capture_slots=1,
        kv_pages=64,
        workspace_bytes=128 * 1024 * 1024,
      ),
    )
  elif gpu_class in {NariGPUClass.RTX_3090, NariGPUClass.RTX_4090}:
    primary_config = optimized if gpu_class is NariGPUClass.RTX_4090 else config
    candidates = (
      _packaged_candidate(f"{gpu_class.value}-b8", "rtx_24gb", primary_config),
      _bounded_candidate(
        f"{gpu_class.value}-b4",
        4,
        safe_fallback,
        capture_slots=1,
        kv_pages=96,
        workspace_bytes=128 * 1024 * 1024,
      ),
    )
  elif gpu_class is NariGPUClass.RTX_5090:
    candidates = (
      _packaged_candidate("blackwell32-b32", "ttfa", optimized),
      _bounded_candidate(
        "blackwell32-b24",
        24,
        optimized,
        capture_slots=2,
        kv_pages=128,
        workspace_bytes=256 * 1024 * 1024,
      ),
      _bounded_candidate(
        "blackwell32-b16",
        16,
        optimized,
        capture_slots=2,
        kv_pages=128,
        workspace_bytes=256 * 1024 * 1024,
      ),
      _bounded_candidate(
        "blackwell32-b12",
        12,
        optimized,
        capture_slots=2,
        kv_pages=128,
        workspace_bytes=256 * 1024 * 1024,
      ),
      _packaged_candidate("blackwell32-b8", "rtx_24gb", safe_fallback),
    )
  else:
    if gpu_class is NariGPUClass.H100:
      prefix = "h100-base"
      primary_config = optimized
    elif gpu_class is NariGPUClass.RTX_PRO_6000:
      prefix = "rtxpro96"
      primary_config = optimized
    else:
      prefix = "large-gpu"
      primary_config = config
    candidates = (
      _bounded_candidate(
        f"{prefix}-b24",
        24,
        primary_config,
        capture_slots=4,
        kv_pages=256,
        workspace_bytes=512 * 1024 * 1024,
      ),
      _bounded_candidate(
        f"{prefix}-b16",
        16,
        primary_config,
        capture_slots=4,
        kv_pages=256,
        workspace_bytes=512 * 1024 * 1024,
      ),
      _bounded_candidate(
        f"{prefix}-b12",
        12,
        primary_config,
        capture_slots=4,
        kv_pages=256,
        workspace_bytes=512 * 1024 * 1024,
      ),
      _packaged_candidate(f"{prefix}-b8", "rtx_24gb", safe_fallback),
    )
    if gpu_class is NariGPUClass.RTX_PRO_6000:
      candidates = (
        _packaged_candidate("rtxpro96-b32", "ttfa", primary_config),
        *candidates,
      )
  return NariPolicyPlan(
    name=f"{gpu_class.value}-{config.policy_mode}",
    gpu_class=gpu_class,
    device_name=device_name,
    memory_in_gb=memory_in_gb,
    candidates=candidates,
  )


def is_capture_startup_failure(error: BaseException) -> bool:
  """Return whether a startup error is eligible for a smaller capture retry."""
  detail = f"{type(error).__name__}: {error}".casefold()
  return any(marker in detail for marker in _CAPTURE_FAILURE_MARKERS)


def build_with_capture_fallback(
  plan: NariPolicyPlan,
  builder: NariEngineBuilder,
  recover: CaptureRecovery,
) -> NariEngineBuild:
  """Build the first working capture profile and preserve unrelated failures."""
  for index, candidate in enumerate(plan.candidates):
    try:
      engine = builder(candidate)
    except Exception as error:
      final = index == len(plan.candidates) - 1
      if final or not is_capture_startup_failure(error):
        raise
      LOGGER.warning(
        "Nari capture profile %s failed; trying %s: %s",
        candidate.name,
        plan.candidates[index + 1].name,
        error,
      )
      recover()
      continue
    LOGGER.info(
      "Resolved Nari policy %s with capture profile %s (%s)",
      plan.name,
      candidate.name,
      candidate.profile.sha256,
    )
    return NariEngineBuild(
      engine=engine,
      runtime_config=candidate.runtime_config,
      policy_name=plan.name,
      capture_profile=candidate.name,
      capture_profile_sha256=candidate.profile.sha256,
      capture_fallbacks=index,
    )
  raise RuntimeError("Nari capture policy did not contain a buildable candidate")


def build_nari_engine_with_policy(
  model: Any,
  assets: Any,
  config: NariRuntimeConfig,
  *,
  model_id: str | None,
  torch_module: Any,
  device: Any,
) -> NariEngineBuild:
  """Resolve the local GPU and construct Nari through its ordered fallback ladder."""
  from nari_qwen3_tts.config import EngineConfig
  from nari_qwen3_tts.factory import build_qwen3_tts_engine

  properties = torch_module.cuda.get_device_properties(device)
  device_name = str(properties.name)
  memory_in_gb = round(int(properties.total_memory) / (1024**3))
  plan = resolve_nari_policy(
    config,
    device_name=device_name,
    memory_in_gb=memory_in_gb,
    model_id=model_id,
  )
  LOGGER.info(
    "Selected Nari policy %s for %s (%d GiB); candidates=%s",
    plan.name,
    plan.device_name,
    plan.memory_in_gb,
    ",".join(candidate.name for candidate in plan.candidates),
  )

  def build(candidate: NariCaptureCandidate) -> Any:
    """Build one captured engine with candidate-specific admission capacity."""
    return build_qwen3_tts_engine(
      model,
      assets,
      config=candidate.profile,
      engine_config=EngineConfig(
        max_active_requests=candidate.runtime_config.active_limit(model_id)
      ),
      capture=True,
    )

  def recover() -> None:
    """Release graph allocations left by a recoverable failed capture attempt."""
    gc.collect()
    torch_module.cuda.empty_cache()

  return build_with_capture_fallback(plan, build, recover)


__all__ = [
  "NariCaptureCandidate",
  "NariEngineBuild",
  "NariGPUClass",
  "NariPolicyPlan",
  "build_nari_engine_with_policy",
  "build_with_capture_fallback",
  "classify_nari_gpu",
  "is_capture_startup_failure",
  "resolve_nari_policy",
]
