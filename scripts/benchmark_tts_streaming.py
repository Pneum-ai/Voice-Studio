"""Benchmark Voice Open streaming with fixed concurrency or Nari-style open-loop load."""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import random
import subprocess
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from scripts.tts_benchmark_metrics import (
  AudibleOnsetDetector,
  AudioChunkObservation,
  PlaybackResult,
  distribution,
  evaluate_playback,
  pcm_duration_ms,
)


@dataclass(frozen=True, slots=True)
class Measurement:
  """Hold client-observed timing and playback results for one request."""

  request_id: str
  scheduled_ms: float
  dispatch_ms: float | None
  scheduling_lag_ms: float | None
  started_ms: float | None
  completed_ms: float | None
  playback: PlaybackResult | None
  chunks: int
  native_streaming: bool | None
  diagnostics: dict[str, float | bool] | None
  error_code: str | None
  error: str | None

  @property
  def real_time_factor(self) -> float | None:
    """Return request wall time divided by generated audio duration."""
    if self.completed_ms is None or self.playback is None:
      return None
    if self.playback.audio_seconds <= 0:
      return None
    return self.completed_ms / 1000 / self.playback.audio_seconds

  def record(self) -> dict[str, Any]:
    """Return a JSON-safe request record with the derived real-time factor."""
    return asdict(self) | {"real_time_factor": self.real_time_factor}


class InFlightTracker:
  """Track client-observed dispatched requests and their peak concurrency."""

  def __init__(self) -> None:
    """Initialize an empty request cohort."""
    self.current = 0
    self.peak = 0

  def start(self) -> None:
    """Record one dispatched request."""
    self.current += 1
    self.peak = max(self.peak, self.current)

  def finish(self) -> None:
    """Retire one dispatched request."""
    self.current -= 1


def _parser() -> argparse.ArgumentParser:
  """Build the benchmark command-line parser."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--url", default="ws://127.0.0.1:13337/v1/speech/stream")
  voices = parser.add_mutually_exclusive_group(required=True)
  voices.add_argument("--profile-id")
  voices.add_argument("--speaker", help="CustomVoice built-in speaker")
  parser.add_argument("--instructions", help="CustomVoice delivery instructions")
  parser.add_argument("--model", default="qwen-tts-1.7B")
  parser.add_argument("--language", default="en")
  parser.add_argument(
    "--text",
    default="This sentence measures time to first audio and sustained synthesis speed.",
  )
  parser.add_argument("--dataset", type=Path, help="Text or JSONL prompts with a text field")
  parser.add_argument("--concurrency", type=int, default=1)
  parser.add_argument("--rounds", type=int, default=3)
  parser.add_argument("--rps", type=float, nargs="+", help="Enable open-loop Poisson load")
  parser.add_argument("--warmup", type=float, default=30.0)
  parser.add_argument("--duration", type=float, default=60.0)
  parser.add_argument("--timeout", type=float, default=120.0)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--output", type=Path)
  parser.add_argument("--no-request-details", action="store_true")
  return parser


def _prompts(args: argparse.Namespace) -> list[str]:
  """Load a deterministic prompt pool from text or JSONL input."""
  if args.dataset is None:
    return [args.text]
  prompts: list[str] = []
  for raw_line in args.dataset.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line:
      continue
    if args.dataset.suffix.casefold() == ".jsonl":
      value = json.loads(line)
      text = value.get("text") if isinstance(value, dict) else None
      if not isinstance(text, str) or not text.strip():
        raise ValueError("Every JSONL row must contain a non-empty text field")
      prompts.append(text.strip())
    else:
      prompts.append(line)
  if not prompts:
    raise ValueError("The benchmark prompt dataset is empty")
  return prompts


def _http_base_url(websocket_url: str) -> str:
  """Convert the speech WebSocket URL into its server HTTP origin."""
  parsed = urlsplit(websocket_url)
  scheme = "https" if parsed.scheme == "wss" else "http"
  return urlunsplit((scheme, parsed.netloc, "", "", ""))


def _command_output(arguments: list[str]) -> str | None:
  """Return one short diagnostic command result when the executable exists."""
  try:
    result = subprocess.run(
      arguments,
      capture_output=True,
      check=True,
      text=True,
      timeout=5,
    )
  except (OSError, subprocess.SubprocessError):
    return None
  return result.stdout.strip() or None


async def _environment(args: argparse.Namespace) -> dict[str, Any]:
  """Collect server identity, model state, revision, and local GPU metadata."""
  import httpx

  base_url = _http_base_url(args.url)
  async with httpx.AsyncClient(timeout=10) as client:
    health, model = await asyncio.gather(
      client.get(f"{base_url}/health"),
      client.get(f"{base_url}/v1/models/{args.model}"),
    )
  health.raise_for_status()
  model.raise_for_status()
  gpu = _command_output(
    [
      "nvidia-smi",
      "--query-gpu=name,memory.total,driver_version,power.limit",
      "--format=csv,noheader,nounits",
    ]
  )
  return {
    "server": health.json(),
    "model": model.json(),
    "client": {
      "platform": platform.platform(),
      "machine": platform.machine(),
      "python": sys.version.split()[0],
    },
    "gpu": gpu.splitlines() if gpu else [],
    "git_revision": _command_output(["git", "rev-parse", "HEAD"]),
    "git_dirty": bool(_command_output(["git", "status", "--porcelain"])),
  }


async def _measure_one(
  args: argparse.Namespace,
  *,
  index: int,
  run_start: float,
  scheduled_at: float,
  text: str,
  tracker: InFlightTracker,
) -> Measurement:
  """Execute one scheduled WebSocket request and simulate immediate playback."""
  import websockets

  request_id = f"bench-{index}-{uuid.uuid4().hex}"
  payload: dict[str, Any] = {
    "type": "generate",
    "request_id": request_id,
    "profile_id": args.profile_id,
    "speaker": args.speaker,
    "instructions": args.instructions,
    "text": text,
    "language": args.language,
    "model": args.model,
    "store": False,
    "diagnostics": True,
  }
  await asyncio.sleep(max(0.0, scheduled_at - time.perf_counter()))
  dispatch_at: float | None = None
  started_at: float | None = None
  completed_at: float | None = None
  chunks: list[AudioChunkObservation] = []
  detector = AudibleOnsetDetector()
  native_streaming: bool | None = None
  error_code: str | None = None
  error: str | None = None
  diagnostics: dict[str, float | bool] | None = None
  try:
    async with asyncio.timeout(args.timeout):
      async with websockets.connect(args.url, max_size=None) as websocket:
        dispatch_at = time.perf_counter()
        tracker.start()
        await websocket.send(json.dumps(payload))
        while True:
          message = await websocket.recv()
          now = time.perf_counter()
          if isinstance(message, bytes):
            detector.feed(message)
            chunks.append(
              AudioChunkObservation(
                arrival_ms=(now - dispatch_at) * 1000,
                duration_ms=pcm_duration_ms(message),
              )
            )
            continue
          event = json.loads(message)
          if event.get("type") == "started":
            started_at = now
            native_streaming = bool(event.get("native_streaming"))
          elif event.get("type") == "finished":
            raw_diagnostics = event.get("diagnostics")
            if isinstance(raw_diagnostics, dict):
              diagnostics = {
                str(key): value
                for key, value in raw_diagnostics.items()
                if isinstance(value, (bool, int, float))
              }
            completed_at = now
            break
          elif event.get("type") == "error":
            error_code = str(event.get("code") or "websocket_error")
            raise RuntimeError(event.get("detail") or event.get("title") or str(event))
  except Exception as caught:
    error_code = error_code or type(caught).__name__
    error = f"{type(caught).__name__}: {caught}"
  finally:
    if dispatch_at is not None:
      tracker.finish()
  playback = evaluate_playback(chunks, audible_offset_ms=detector.onset_ms) if chunks else None
  return Measurement(
    request_id=request_id,
    scheduled_ms=(scheduled_at - run_start) * 1000,
    dispatch_ms=(dispatch_at - run_start) * 1000 if dispatch_at is not None else None,
    scheduling_lag_ms=(dispatch_at - scheduled_at) * 1000 if dispatch_at is not None else None,
    started_ms=(started_at - dispatch_at) * 1000 if started_at and dispatch_at else None,
    completed_ms=(completed_at - dispatch_at) * 1000 if completed_at and dispatch_at else None,
    playback=playback,
    chunks=len(chunks),
    native_streaming=native_streaming,
    diagnostics=diagnostics,
    error_code=error_code,
    error=error,
  )


def _summary(
  measurements: list[Measurement],
  *,
  requested_rps: float | None,
  actual_rps: float | None,
  peak_in_flight: int,
) -> dict[str, Any]:
  """Aggregate one cohort using Nari-compatible nearest-rank percentiles."""
  successful = [item for item in measurements if item.error is None and item.playback]
  underrun_requests = [item for item in successful if item.playback and item.playback.underruns]
  diagnostic_keys = sorted(
    {
      key
      for item in successful
      for key, value in (item.diagnostics or {}).items()
      if not isinstance(value, bool)
    }
  )
  cache_results = [
    item.diagnostics["prompt_cache_hit"]
    for item in successful
    if item.diagnostics is not None and isinstance(item.diagnostics.get("prompt_cache_hit"), bool)
  ]
  return {
    "requested_rps": requested_rps,
    "actual_rps": actual_rps,
    "requests": len(measurements),
    "successful_requests": len(successful),
    "failed_requests": len(measurements) - len(successful),
    "failure_codes": dict(
      sorted(Counter(item.error_code for item in measurements if item.error_code).items())
    ),
    "native_streaming": sorted(
      {item.native_streaming for item in measurements if item.native_streaming is not None}
    ),
    "peak_in_flight": peak_in_flight,
    "underrun_requests": len(underrun_requests),
    "underrun_events": sum(
      item.playback.underruns for item in successful if item.playback is not None
    ),
    "first_playable_ms": distribution(
      [item.playback.first_playable_ms if item.playback else None for item in measurements]
    ),
    "audible_ttfa_ms": distribution(
      [item.playback.audible_ttfa_ms if item.playback else None for item in measurements]
    ),
    "leading_silence_ms": distribution(
      [item.playback.leading_silence_ms if item.playback else None for item in measurements]
    ),
    "end_to_end_ms": distribution([item.completed_ms for item in measurements]),
    "audio_duration_s": distribution(
      [item.playback.audio_seconds if item.playback else None for item in measurements]
    ),
    "real_time_factor": distribution([item.real_time_factor for item in measurements]),
    "scheduling_lag_ms": distribution([item.scheduling_lag_ms for item in measurements]),
    "engine_diagnostics": {
      key: distribution(
        [
          float(item.diagnostics[key])
          if item.diagnostics is not None and key in item.diagnostics
          else None
          for item in measurements
        ]
      )
      for key in diagnostic_keys
    },
    "prompt_cache_hit_rate": (
      sum(bool(value) for value in cache_results) / len(cache_results) if cache_results else None
    ),
  }


def _poisson_schedule(rps: float, duration: float, seed: int) -> list[float]:
  """Create deterministic open-loop arrival offsets for one target rate."""
  rng = random.Random(seed)
  offsets: list[float] = []
  offset = 0.0
  while True:
    offset += rng.expovariate(rps)
    if offset >= duration:
      return offsets
    offsets.append(offset)


async def _run_open_loop(
  args: argparse.Namespace,
  *,
  rps: float,
  prompts: list[str],
  seed: int,
) -> dict[str, Any]:
  """Run warm-up and measurement phases with Poisson open-loop arrivals."""
  total_duration = args.warmup + args.duration
  offsets = _poisson_schedule(rps, total_duration, seed)
  prompt_rng = random.Random(seed + 1)
  run_start = time.perf_counter()
  tracker = InFlightTracker()
  tasks = [
    _measure_one(
      args,
      index=index,
      run_start=run_start,
      scheduled_at=run_start + offset,
      text=prompts[prompt_rng.randrange(len(prompts))],
      tracker=tracker,
    )
    for index, offset in enumerate(offsets)
  ]
  all_measurements = list(await asyncio.gather(*tasks))
  measurements = [item for item in all_measurements if item.scheduled_ms >= args.warmup * 1000]
  result: dict[str, Any] = {
    "summary": _summary(
      measurements,
      requested_rps=rps,
      actual_rps=len(measurements) / args.duration,
      peak_in_flight=tracker.peak,
    )
  }
  if not args.no_request_details:
    result["request_records"] = [item.record() for item in measurements]
  return result


async def _run_fixed(args: argparse.Namespace, prompts: list[str]) -> dict[str, Any]:
  """Retain the original rounds-by-concurrency smoke benchmark mode."""
  tracker = InFlightTracker()
  measurements: list[Measurement] = []
  prompt_rng = random.Random(args.seed)
  run_start = time.perf_counter()
  for round_index in range(args.rounds):
    scheduled_at = time.perf_counter()
    offset = round_index * args.concurrency
    values = await asyncio.gather(
      *(
        _measure_one(
          args,
          index=offset + index,
          run_start=run_start,
          scheduled_at=scheduled_at,
          text=prompts[prompt_rng.randrange(len(prompts))],
          tracker=tracker,
        )
        for index in range(args.concurrency)
      )
    )
    measurements.extend(values)
  result: dict[str, Any] = {
    "summary": _summary(
      measurements,
      requested_rps=None,
      actual_rps=None,
      peak_in_flight=tracker.peak,
    )
  }
  if not args.no_request_details:
    result["request_records"] = [item.record() for item in measurements]
  return result


def _validate(args: argparse.Namespace) -> None:
  """Reject invalid benchmark timing and load values before opening sockets."""
  if args.concurrency < 1 or args.rounds < 1:
    raise ValueError("concurrency and rounds must be positive")
  if args.warmup < 0 or args.duration <= 0 or args.timeout <= 0:
    raise ValueError("warmup must be nonnegative; duration and timeout must be positive")
  if args.rps is not None and any(value <= 0 for value in args.rps):
    raise ValueError("every RPS value must be positive")


async def _run(args: argparse.Namespace) -> dict[str, Any]:
  """Execute the selected benchmark modes and collect reproducibility metadata."""
  _validate(args)
  prompts = _prompts(args)
  environment = await _environment(args)
  if args.rps is None:
    runs = [await _run_fixed(args, prompts)]
    mode = "fixed_concurrency"
  else:
    runs = [
      await _run_open_loop(args, rps=rps, prompts=prompts, seed=args.seed + index)
      for index, rps in enumerate(args.rps)
    ]
    mode = "poisson_open_loop"
  return {
    "schema_version": 3,
    "mode": mode,
    "model": args.model,
    "profile_id": args.profile_id,
    "speaker": args.speaker,
    "instructions": args.instructions,
    "prompt_count": len(prompts),
    "warmup_seconds": args.warmup if args.rps else 0,
    "duration_seconds": args.duration if args.rps else None,
    "seed": args.seed,
    "environment": environment,
    "measurement_contract": {
      "transport": "WebSocket PCM16 mono 24 kHz",
      "timing_origin": "immediately before generate payload send",
      "arrival": "Poisson open-loop" if args.rps else "fixed concurrent rounds",
      "audible_detector": {
        "frame_ms": AudibleOnsetDetector.frame_ms,
        "hop_ms": AudibleOnsetDetector.hop_ms,
        "threshold_dbfs": AudibleOnsetDetector.threshold_dbfs,
        "consecutive_active_frames": AudibleOnsetDetector.consecutive_active_frames,
      },
    },
    "runs": runs,
  }


def main() -> None:
  """Run the benchmark and print or persist machine-readable JSON."""
  args = _parser().parse_args()
  result = asyncio.run(_run(args))
  encoded = json.dumps(result, indent=2, sort_keys=True)
  if args.output is not None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")
  print(encoded)


if __name__ == "__main__":
  main()
