"""Per-request stage timings for diagnosing synthesis latency."""

from __future__ import annotations

from dataclasses import dataclass, field

from nari_qwen3_tts.contract.stage import SynthesisStage


@dataclass(frozen=True, slots=True)
class SynthesisTimingSnapshot:
    """Expose stable millisecond offsets from engine admission."""

    model_prepare_ms: float
    prompt_prepare_ms: float
    prompt_cache_hit: bool
    first_dispatch_ms: tuple[tuple[str, float], ...]
    first_commit_ms: tuple[tuple[str, float], ...]
    first_pcm_ms: float | None
    total_engine_ms: float

    def as_dict(self) -> dict[str, float | bool]:
        """Flatten stage offsets into JSON-safe benchmark fields."""
        result: dict[str, float | bool] = {
            "model_prepare_ms": self.model_prepare_ms,
            "prompt_prepare_ms": self.prompt_prepare_ms,
            "prompt_cache_hit": self.prompt_cache_hit,
            "total_engine_ms": self.total_engine_ms,
        }
        if self.first_pcm_ms is not None:
            result["first_pcm_ms"] = self.first_pcm_ms
        for stage, value in self.first_dispatch_ms:
            result[f"{stage}_dispatch_ms"] = value
        for stage, value in self.first_commit_ms:
            result[f"{stage}_commit_ms"] = value
        return result


@dataclass(slots=True)
class SynthesisTimingRecorder:
    """Collect first-occurrence stage timestamps on the engine thread."""

    admitted_at_s: float
    model_prepare_ms: float
    prompt_prepare_ms: float
    prompt_cache_hit: bool
    first_dispatch_at_s: dict[SynthesisStage, float] = field(default_factory=dict)
    first_commit_at_s: dict[SynthesisStage, float] = field(default_factory=dict)
    first_pcm_at_s: float | None = None

    def record_dispatch(self, stage: SynthesisStage, *, at_s: float) -> None:
        """Record the first GPU submission for a synthesis stage."""
        self.first_dispatch_at_s.setdefault(stage, at_s)

    def record_commit(self, stage: SynthesisStage, *, at_s: float) -> None:
        """Record the first committed completion for a synthesis stage."""
        self.first_commit_at_s.setdefault(stage, at_s)

    def record_pcm(self, *, at_s: float) -> None:
        """Record the first PCM crossing into the client delivery queue."""
        if self.first_pcm_at_s is None:
            self.first_pcm_at_s = at_s

    def snapshot(self, *, finished_at_s: float) -> SynthesisTimingSnapshot:
        """Freeze current timing offsets for delivery to diagnostics clients."""
        offset = self.admitted_at_s
        return SynthesisTimingSnapshot(
            model_prepare_ms=self.model_prepare_ms,
            prompt_prepare_ms=self.prompt_prepare_ms,
            prompt_cache_hit=self.prompt_cache_hit,
            first_dispatch_ms=tuple(
                (stage.value, (at_s - offset) * 1000)
                for stage, at_s in self.first_dispatch_at_s.items()
            ),
            first_commit_ms=tuple(
                (stage.value, (at_s - offset) * 1000)
                for stage, at_s in self.first_commit_at_s.items()
            ),
            first_pcm_ms=(
                None if self.first_pcm_at_s is None else (self.first_pcm_at_s - offset) * 1000
            ),
            total_engine_ms=(finished_at_s - offset) * 1000,
        )


__all__ = ["SynthesisTimingRecorder", "SynthesisTimingSnapshot"]
