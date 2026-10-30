"""Batch assembly, grouping, and cancellation for concurrent speech requests."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import numpy as np
import pytest

from app.backends.factory import BackendBundle
from app.backends.protocols import BatchGenerationRequest
from app.services.tts_batcher import TTSBatcher
from app.utils.audio import AudioData


class RecordingTTSBackend:
  """Record every batch without running a model."""

  def __init__(self, *, delay: float = 0.0, concurrent_streaming: bool = False) -> None:
    """Start with no recorded batches."""
    self.batches: list[list[str]] = []
    self.seeds: list[int | None] = []
    self.delay = delay
    self.failure: Exception | None = None
    self.concurrent_streaming = concurrent_streaming

  @property
  def supports_streaming(self) -> bool:
    """Report a complete-waveform backend so requests are batched."""
    return False

  @property
  def supports_concurrent_streaming(self) -> bool:
    """Report whether compatible native streams may share the selected runtime."""
    return self.concurrent_streaming

  async def generate_batch(
    self,
    requests: Sequence[BatchGenerationRequest],
    *,
    seed: int | None,
    cancel_events: Sequence[asyncio.Event | None] | None = None,
  ) -> list[AudioData]:
    """Record one batch and return a distinct waveform per request."""
    _ = cancel_events
    self.batches.append([request.text for request in requests])
    self.seeds.append(seed)
    if self.delay:
      await asyncio.sleep(self.delay)
    if self.failure is not None:
      raise self.failure
    return [
      AudioData(np.array([float(len(request.text))], dtype=np.float32), 24_000)
      for request in requests
    ]


def build_batcher(
  backend: RecordingTTSBackend,
  **kwargs: object,
) -> TTSBatcher:
  """Create a batcher over a recording backend with test-sized windows."""
  bundle = BackendBundle(
    name="fake",
    device="cpu",
    tts=backend,  # type: ignore[arg-type]
    stt=backend,  # type: ignore[arg-type]
  )
  return TTSBatcher(bundle, **kwargs)  # type: ignore[arg-type]


def request_for(text: str) -> BatchGenerationRequest:
  """Build a minimal resolved request for one piece of text."""
  return BatchGenerationRequest(text=text, prompt=["prompt"], language="en", instructions=None)


async def test_backend_owned_queue_joins_live_stream_without_outer_batch_delay() -> None:
  """Saved requests must enter the same backend while an existing live stream holds a lease."""
  entered = asyncio.Event()
  release = asyncio.Event()

  class ManagedBackend(RecordingTTSBackend):
    """Expose backend-owned scheduling while retaining observable request correlation."""

    manages_request_queue = True

    async def generate_batch(
      self,
      requests: Sequence[BatchGenerationRequest],
      *,
      seed: int | None,
      cancel_events: Sequence[asyncio.Event | None] | None = None,
    ) -> list[AudioData]:
      """Signal entry before waiting so the test detects a conflicting outer lease."""
      entered.set()
      await release.wait()
      return await super().generate_batch(requests, seed=seed, cancel_events=cancel_events)

  backend = ManagedBackend(concurrent_streaming=True)
  batcher = build_batcher(backend, window_seconds=10)
  try:
    async with batcher.streaming_section("qwen-tts-0.6B"):
      saved = asyncio.create_task(submit(batcher, "Saved"))
      await asyncio.wait_for(entered.wait(), 0.5)
      release.set()
      assert (await saved).samples.tolist() == [5]
  finally:
    release.set()
    await batcher.stop()


async def submit(
  batcher: TTSBatcher,
  text: str,
  *,
  model_id: str = "qwen-tts-0.6B",
  seed: int | None = None,
  cancel_event: asyncio.Event | None = None,
) -> AudioData:
  """Submit one request with the defaults used across these tests."""
  return await batcher.submit(
    model_id=model_id,
    request=request_for(text),
    seed=seed,
    cancel_event=cancel_event,
  )


async def test_batch_size_limit_is_respected() -> None:
  """Stop collecting once the configured maximum batch size is reached."""
  backend = RecordingTTSBackend()
  batcher = build_batcher(backend, max_batch_size=2, window_seconds=0.05)
  await asyncio.gather(*(submit(batcher, f"chunk number {index}") for index in range(4)))
  await batcher.stop()
  assert [len(batch) for batch in backend.batches] == [2, 2]


async def test_requests_with_different_models_do_not_share_a_pass() -> None:
  """Keep distinct loaded models in separate backend passes."""
  backend = RecordingTTSBackend()
  batcher = build_batcher(backend, max_batch_size=8, window_seconds=0.05)
  await asyncio.gather(
    submit(batcher, "small model line"),
    submit(batcher, "large model line", model_id="qwen-tts-1.7B"),
  )
  await batcher.stop()
  assert backend.batches == [["small model line"], ["large model line"]]


async def test_length_mismatch_starts_a_new_batch() -> None:
  """Avoid pairing a short request with one that runs far longer."""
  backend = RecordingTTSBackend()
  batcher = build_batcher(backend, max_batch_size=8, window_seconds=0.05)
  await asyncio.gather(
    submit(batcher, "short"),
    submit(batcher, "a considerably longer chunk of speech than the first one"),
  )
  await batcher.stop()
  assert [len(batch) for batch in backend.batches] == [1, 1]


async def test_seeded_request_runs_alone() -> None:
  """Run seeded requests by themselves so their seed stays reproducible."""
  backend = RecordingTTSBackend()
  batcher = build_batcher(backend, max_batch_size=8, window_seconds=0.05)
  await asyncio.gather(
    submit(batcher, "seeded line here", seed=7),
    submit(batcher, "another line here"),
  )
  await batcher.stop()
  assert [len(batch) for batch in backend.batches] == [1, 1]
  assert backend.seeds == [7, None]


async def test_cancelled_request_is_dropped_before_dispatch() -> None:
  """Skip work for a request whose caller cancelled before its batch ran."""
  backend = RecordingTTSBackend(delay=0.05)
  batcher = build_batcher(backend, max_batch_size=1, window_seconds=0.0)
  cancelled = asyncio.Event()
  blocker = asyncio.create_task(submit(batcher, "first line of speech"))
  await asyncio.sleep(0)
  dropped = asyncio.create_task(submit(batcher, "second line", cancel_event=cancelled))
  cancelled.set()
  await blocker
  dropped.cancel()
  await asyncio.gather(dropped, return_exceptions=True)
  await batcher.stop()
  assert backend.batches == [["first line of speech"]]


async def test_backend_failure_reaches_every_caller_in_the_batch() -> None:
  """Propagate one generation failure to each request that shared the pass."""
  backend = RecordingTTSBackend()
  backend.failure = RuntimeError("generation exploded")
  batcher = build_batcher(backend, max_batch_size=8, window_seconds=0.05)
  results = await asyncio.gather(
    submit(batcher, "first line here"),
    submit(batcher, "second line here"),
    return_exceptions=True,
  )
  await batcher.stop()
  assert all(isinstance(result, RuntimeError) for result in results)


async def test_batching_can_be_disabled() -> None:
  """Fall back to one request per pass when batching is switched off."""
  backend = RecordingTTSBackend()
  batcher = build_batcher(backend, enabled=False, max_batch_size=8, window_seconds=0.05)
  await asyncio.gather(*(submit(batcher, f"chunk number {index}") for index in range(3)))
  await batcher.stop()
  assert [len(batch) for batch in backend.batches] == [1, 1, 1]


async def test_stop_fails_requests_that_never_ran() -> None:
  """Report shutdown to callers still waiting when the batcher stops."""
  backend = RecordingTTSBackend(delay=0.2)
  batcher = build_batcher(backend, max_batch_size=1, window_seconds=0.0)
  first = asyncio.create_task(submit(batcher, "first line of speech"))
  await asyncio.sleep(0)
  queued = asyncio.create_task(submit(batcher, "queued line of speech"))
  await asyncio.sleep(0.01)
  await batcher.stop()
  with pytest.raises(RuntimeError, match="shutting down"):
    await queued
  first.cancel()
  await asyncio.gather(first, return_exceptions=True)


async def test_same_model_native_streams_overlap() -> None:
  """Let Nari schedule concurrent streams that share one loaded model."""
  backend = RecordingTTSBackend(concurrent_streaming=True)
  batcher = build_batcher(backend)
  both_entered = asyncio.Event()
  release = asyncio.Event()
  entered = 0

  async def stream() -> None:
    """Hold one compatible stream lease until both callers overlap."""
    nonlocal entered
    async with batcher.streaming_section("qwen-tts-0.6B"):
      entered += 1
      if entered == 2:
        both_entered.set()
      await release.wait()

  tasks = [asyncio.create_task(stream()) for _ in range(2)]
  await asyncio.wait_for(both_entered.wait(), timeout=1)
  release.set()
  await asyncio.gather(*tasks)


async def test_different_model_native_streams_serialize() -> None:
  """Keep a loaded Nari runtime alive until every stream using it completes."""
  backend = RecordingTTSBackend(concurrent_streaming=True)
  batcher = build_batcher(backend)
  first_entered = asyncio.Event()
  release_first = asyncio.Event()
  second_entered = asyncio.Event()

  async def stream(model_id: str, entered: asyncio.Event, release: asyncio.Event) -> None:
    """Record entry while holding one model-specific stream lease."""
    async with batcher.streaming_section(model_id):
      entered.set()
      await release.wait()

  first = asyncio.create_task(stream("qwen-tts-0.6B", first_entered, release_first))
  await first_entered.wait()
  release_second = asyncio.Event()
  second = asyncio.create_task(stream("qwen-tts-1.7B", second_entered, release_second))
  await asyncio.sleep(0.01)
  assert not second_entered.is_set()
  release_first.set()
  await asyncio.wait_for(second_entered.wait(), timeout=1)
  release_second.set()
  await asyncio.gather(first, second)
