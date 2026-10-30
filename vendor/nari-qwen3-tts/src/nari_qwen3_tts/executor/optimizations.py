"""Post-load FP8 weight preparation for captured Qwen3-TTS execution.

Model forward methods own device- and dtype-aware kernel dispatch. This module
prepares Talker gate/up weights as FP8 blocks and enables optimized model math
after exact checkpoint loading.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

import torch

from nari_qwen3_tts.model.layers import GatedMLP

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecutionOptimizationReport:
    talker_mlp_layers: int
    talker_fp8_layers: int
    code_predictor_mlp_layers: int
    talker_fp8_backend: str | None
    blackwell_speedup: float | None = None
    blackwell_relative_error: float | None = None


def _elapsed_projection_ms(layer: GatedMLP, inputs: torch.Tensor, *, fp8: bool) -> float:
    """Measure one projection path using CUDA events after warm-up."""

    operation = layer._project_gate_up if fp8 else layer.gate_up_proj
    for _ in range(10):
        operation(inputs)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        operation(inputs)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / 100


def _benchmark_blackwell(layer: GatedMLP) -> tuple[float, float]:
    """Return BF16-over-FP8 speedup and relative mean projection error."""

    inputs = torch.linspace(
        -1.0,
        1.0,
        steps=layer.gate_up_proj.input_size,
        device=layer.gate_up_proj.weight.device,
        dtype=torch.bfloat16,
    ).view(1, -1)
    reference = layer.gate_up_proj(inputs).float()
    candidate = layer._project_gate_up(inputs).float()
    denominator = reference.abs().mean().clamp_min(1e-6)
    relative_error = float((reference - candidate).abs().mean() / denominator)
    bf16_ms = _elapsed_projection_ms(layer, inputs, fp8=False)
    fp8_ms = _elapsed_projection_ms(layer, inputs, fp8=True)
    return bf16_ms / fp8_ms, relative_error


def _install_blackwell_fp8(
    layers: tuple[object, ...],
    *,
    mode: Literal["off", "auto", "force"],
) -> tuple[int, float | None, float | None]:
    """Enable scaled GEMM only when forced or faster with bounded projection error."""

    if mode == "off" or not layers:
        return 0, None, None
    first = layers[0].mlp
    if not first.initialize_fp8_gate_up_weight(blackwell=True):
        return 0, None, None
    try:
        speedup, relative_error = _benchmark_blackwell(first)
    except Exception:
        first.disable_fp8_gate_up_weight()
        if mode == "force":
            raise
        LOGGER.warning(
            "Blackwell FP8 probe failed; retaining the BF16 Talker path",
            exc_info=True,
        )
        return 0, None, None
    accepted = mode == "force" or (speedup >= 1.05 and relative_error <= 0.05)
    if not accepted:
        first.disable_fp8_gate_up_weight()
        LOGGER.info(
            "Rejected Blackwell FP8 Talker path: speedup=%.3fx relative_error=%.4f",
            speedup,
            relative_error,
        )
        return 0, speedup, relative_error
    initialized = 1 + sum(
        int(layer.mlp.initialize_fp8_gate_up_weight(blackwell=True)) for layer in layers[1:]
    )
    LOGGER.info(
        "Enabled Blackwell FP8 Talker path on %d layers: speedup=%.3fx relative_error=%.4f",
        initialized,
        speedup,
        relative_error,
    )
    return initialized, speedup, relative_error


def install_capture_optimizations(
    assets: object,
    *,
    require_talker_fp8: bool,
) -> ExecutionOptimizationReport:
    """Quantize Talker gate/up weights and report the resulting coverage."""

    talker_layers = assets.talker.model.layers
    cp_layers = assets.code_predictor.model.layers
    for layers in (talker_layers, cp_layers):
        for layer in layers:
            if not isinstance(layer.mlp, GatedMLP):
                raise TypeError("capture optimization requires the Qwen3-TTS GatedMLP")
    capability = torch.cuda.get_device_capability(assets.device)
    blackwell_mode = assets.config.blackwell_fp8 if capability[0] == 12 else "off"
    if blackwell_mode != "off":
        fp8_layers, speedup, relative_error = _install_blackwell_fp8(
            tuple(talker_layers),
            mode=blackwell_mode,
        )
    else:
        fp8_layers = sum(int(layer.mlp.initialize_fp8_gate_up_weight()) for layer in talker_layers)
        speedup = None
        relative_error = None
    if require_talker_fp8 and fp8_layers != len(talker_layers):
        raise RuntimeError(
            f"H100 profile requires FP8 gate/up for every Talker layer: {fp8_layers}/{len(talker_layers)}"
        )
    for layer in (*talker_layers, *cp_layers):
        layer.mlp.enable_cuda_optimized_math()
    return ExecutionOptimizationReport(
        talker_mlp_layers=len(talker_layers),
        talker_fp8_layers=fp8_layers,
        code_predictor_mlp_layers=len(cp_layers),
        talker_fp8_backend=(talker_layers[0].mlp.fp8_backend if fp8_layers else None),
        blackwell_speedup=speedup,
        blackwell_relative_error=relative_error,
    )


__all__ = [
    "ExecutionOptimizationReport",
    "install_capture_optimizations",
]
