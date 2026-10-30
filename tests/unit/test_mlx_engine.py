"""Exercise the real async MLX engine with deterministic CPU inference events."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from app.backends.mlx.engine import SpeechEngine
from app.backends.mlx.inference import Event, Request
from app.backends.mlx.tts import MLXTTSBackend
from app.backends.protocols import BatchGenerationRequest
from app.core.errors import InferenceTimeoutError, InferenceUnavailableError, OverloadedError
from app.utils.audio import AudioData


class CPUInference:
  """Stand in for GPU math while leaving queueing and delivery completely real."""

  def __init__(self, model: SimpleNamespace, model_id: str) -> None:
    """Record each native instance to detect leaked or duplicated scheduler ownership."""
    self.model = model
    self.active: dict[int, tuple[Request, int]] = {}
    self.closed = False
    model.instances.append(self)

  def tick(
    self, incoming: list[Request], cancelled: set[int], feedback: list[tuple[int, float, float]]
  ) -> list[Event]:
    """Emit distinct non-overlapping PCM; gates expose cancellation and backpressure races."""
    del feedback
    events = []
    for request in incoming:
      self.active[request.identity] = (request, 0)
      events.append(Event(request.identity, started=True))
    for identity in cancelled:
      self.active.pop(identity, None)
      self.model.cancelled.add(identity)
      events.append(Event(identity, done=True))
    for identity, (request, index) in list(self.active.items()):
      if index and not self.model.release.is_set():
        time.sleep(0.001)
        continue
      if index and self.model.failure is not None:
        raise self.model.failure
      samples = np.full(24, len(request.text) + index, dtype=np.float32)
      events.append(Event(identity, AudioData(samples, 24000)))
      if index + 1 == self.model.chunks:
        events.append(Event(identity, done=True))
        del self.active[identity]
      else:
        self.active[identity] = (request, index + 1)
    return events

  def close(self) -> None:
    """Expose native cleanup so shutdown and disconnect assertions are meaningful."""
    self.closed = True
    self.active.clear()


@pytest.fixture
async def backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[MLXTTSBackend]:
  """Create the actual delivery engine and GPU worker with controllable CPU inference."""
  monkeypatch.setattr("app.backends.mlx.engine.Inference", CPUInference)
  model = SimpleNamespace(
    instances=[], cancelled=set(), release=threading.Event(), chunks=3, failure=None
  )
  model.release.set()
  backend = MLXTTSBackend(tmp_path)
  backend._model = model
  backend._loaded_model = "qwen-tts-0.6B"
  backend._engine = SpeechEngine(model, backend._loaded_model, backend._worker)
  try:
    yield backend
  finally:
    model.release.set()
    await backend.unload()


def request(text: str) -> BatchGenerationRequest:
  """Construct immutable conditioning for either public speech entry point."""
  return BatchGenerationRequest(
    text, {"ref_audio": "voice.wav", "ref_text": "Reference."}, "en", None
  )


def stream(
  backend: MLXTTSBackend, text: str = "Live", cancel: asyncio.Event | None = None
) -> AsyncGenerator[AudioData, None]:
  """Open a public backend stream without bypassing its unified engine."""
  return backend.stream(
    text=text,
    prompt=request(text).prompt,
    language="en",
    seed=None,
    instructions=None,
    cancel_event=cancel or asyncio.Event(),
  )


async def test_saved_and_live_requests_share_one_scheduler_and_preserve_samples(
  backend: MLXTTSBackend,
) -> None:
  """Saved speech collects the same PCM path while live speech is already playing."""
  model = cast(SimpleNamespace, backend._model)
  model.release.clear()
  live = stream(backend)
  first = await anext(live)
  assert not model.instances[0].closed
  saved = asyncio.create_task(backend.generate_batch([request("Saved")], seed=None))
  for _ in range(100):
    if len(model.instances[0].active) == 2:
      break
    await asyncio.sleep(0.001)
  assert len(model.instances) == 1 and len(model.instances[0].active) == 2
  model.release.set()
  live_audio = np.concatenate([first.samples] + [item.samples async for item in live])
  saved_audio = (await saved)[0]
  np.testing.assert_array_equal(live_audio, np.repeat([4, 5, 6], 24))
  np.testing.assert_array_equal(saved_audio.samples, np.repeat([5, 6, 7], 24))


async def test_batch_cancellation_keeps_other_results_in_order(backend: MLXTTSBackend) -> None:
  """A cancelled slot is empty; surviving slots use the same streamed generation path."""
  cancel = asyncio.Event()
  cancel.set()
  outputs = await backend.generate_batch(
    [request("One"), request("Cancelled"), request("Three")],
    seed=None,
    cancel_events=[None, cancel, None],
  )
  assert outputs[1].samples.size == 0
  np.testing.assert_array_equal(outputs[0].samples, np.repeat([3, 4, 5], 24))
  np.testing.assert_array_equal(outputs[2].samples, np.repeat([5, 6, 7], 24))


async def test_disconnect_releases_native_request(backend: MLXTTSBackend) -> None:
  """Closing a partially consumed stream cancels its native row before returning."""
  model = cast(SimpleNamespace, backend._model)
  model.release.clear()
  iterator = stream(backend)
  await anext(iterator)
  await asyncio.wait_for(iterator.aclose(), 1)
  assert model.cancelled == {0}
  assert not model.instances[0].active


async def test_failure_after_first_audio_is_not_silently_completed(backend: MLXTTSBackend) -> None:
  """A failure must still reach a consumer whose cancellation flag is set internally."""
  model = cast(SimpleNamespace, backend._model)
  model.release.clear()
  iterator = stream(backend)
  await anext(iterator)
  model.failure = ValueError("Decoder failed")
  model.release.set()
  with pytest.raises(ValueError, match="Decoder failed"):
    await anext(iterator)


async def test_slow_consumer_does_not_stall_another_listener(
  backend: MLXTTSBackend,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Bound output memory and explicitly fail a stalled reader while others complete."""
  monkeypatch.setattr("app.backends.mlx.engine.MAX_BUFFERED_CHUNKS", 2)
  cast(SimpleNamespace, backend._model).release.clear()
  cast(SimpleNamespace, backend._model).chunks = 20
  slow = stream(backend, "Slow")
  await anext(slow)

  async def collect() -> int:
    """Drain the healthy reader promptly through the same engine."""
    return sum([item.samples.size async for item in stream(backend, "Healthy")])

  fast = asyncio.create_task(collect())
  await asyncio.sleep(0.01)
  cast(SimpleNamespace, backend._model).release.set()
  assert await asyncio.wait_for(fast, 2) == 480
  with pytest.raises(InferenceTimeoutError, match="audio consumer"):
    await anext(slow)


async def test_queue_limit_rejects_excess_without_losing_existing_work(
  backend: MLXTTSBackend,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Reject overload before allocating another request's native caches."""
  monkeypatch.setattr("app.backends.mlx.engine.MAX_REQUESTS", 1)
  cast(SimpleNamespace, backend._model).release.clear()
  first = stream(backend)
  await anext(first)
  with pytest.raises(OverloadedError):
    await anext(stream(backend, "Excess"))
  cast(SimpleNamespace, backend._model).release.set()
  assert sum([item.samples.size async for item in first]) == 48


async def test_deadline_cleans_up_waiting_native_work(
  backend: MLXTTSBackend,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """A stalled model step loop cannot leave its consumer waiting indefinitely."""
  monkeypatch.setattr("app.backends.mlx.engine.REQUEST_TIMEOUT_SECONDS", 0.02)
  cast(SimpleNamespace, backend._model).release.clear()
  iterator = stream(backend)
  await anext(iterator)
  with pytest.raises(InferenceTimeoutError, match="deadline"):
    await asyncio.wait_for(anext(iterator), 1)
  assert cast(SimpleNamespace, backend._model).cancelled == {0}


async def test_unload_wakes_live_consumers_and_releases_native_state(
  backend: MLXTTSBackend,
) -> None:
  """Direct unload cannot release model weights underneath an outstanding native step."""
  model = cast(SimpleNamespace, backend._model)
  model.release.clear()
  iterator = stream(backend)
  await anext(iterator)
  await asyncio.wait_for(backend.unload(), 1)
  with pytest.raises(InferenceUnavailableError):
    await anext(iterator)
  assert model.instances[0].closed
  assert backend.loaded_model is None


async def test_waiting_transcription_drains_speech_before_admitting_newcomers(
  backend: MLXTTSBackend,
) -> None:
  """Protect ongoing playback while ensuring continuous speech cannot starve STT."""
  model = cast(SimpleNamespace, backend._model)
  model.release.clear()
  playing = stream(backend)
  await anext(playing)
  stt_started = threading.Event()
  stt_release = threading.Event()

  def transcribe() -> None:
    """Hold the worker so a later speech request must wait for STT completion."""
    stt_started.set()
    assert stt_release.wait(2)

  transcription = asyncio.create_task(backend._worker.run(transcribe))
  await asyncio.sleep(0.01)
  later = stream(backend, "Later")
  first_later = asyncio.create_task(anext(later))
  await asyncio.sleep(0.01)
  assert not stt_started.is_set()
  assert len(model.instances[0].active) == 1
  model.release.set()
  assert sum([audio.samples.size async for audio in playing]) == 48
  assert await asyncio.to_thread(stt_started.wait, 1)
  assert not first_later.done()
  stt_release.set()
  await transcription
  assert (await first_later).samples.size == 24
  await later.aclose()


async def test_cancel_queued_behind_transcription_does_not_wait_for_gpu(
  backend: MLXTTSBackend,
) -> None:
  """A request never submitted to the GPU can be cancelled immediately."""
  started = threading.Event()
  release = threading.Event()

  def transcribe() -> None:
    """Occupy the shared worker while a new speech request is still in its async queue."""
    started.set()
    assert release.wait(2)

  busy = asyncio.create_task(backend._worker.run(transcribe))
  assert await asyncio.to_thread(started.wait, 1)
  cancel = asyncio.Event()
  iterator = stream(backend, cancel=cancel)
  pending = asyncio.create_task(anext(iterator))
  await asyncio.sleep(0.01)
  cancel.set()
  try:
    with pytest.raises(StopAsyncIteration):
      await asyncio.wait_for(pending, 0.5)
  finally:
    release.set()
    await busy
    await iterator.aclose()
