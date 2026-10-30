"""Ordered phrase prefetch for incremental Nari Base voice-cloning streams."""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from app.backends.protocols import BatchGenerationRequest
from app.core.errors import ValidationError
from app.utils.audio import StreamAudioData

if TYPE_CHECKING:
  from collections.abc import AsyncIterator

  from app.backends.pytorch.nari_tts import NariTTSRuntime

_MIN_SEGMENT_CHARACTERS = 24
_TARGET_SEGMENT_CHARACTERS = 160
_MAX_SEGMENT_CHARACTERS = 500
_SENTENCE_BOUNDARY = re.compile(r"[.!?](?:[\"')\]]+)?(?:\s+|$)")


@dataclass(frozen=True, slots=True)
class _SegmentFailure:
  """Carry one prefetched segment failure into ordered audio consumption."""

  error: BaseException


@dataclass(slots=True)
class _SegmentOutput:
  """Hold bounded native chunks for one independently generated phrase."""

  queue: asyncio.Queue[StreamAudioData | _SegmentFailure | None]
  task: asyncio.Task[None]


class NariLiveTextSession:
  """Prefetch complete Base phrases and drain their native PCM in text order.

  Qwen3-TTS Base places target text in its ICL prefill, so its captured state
  cannot safely be extended like CustomVoice. Each stable phrase therefore
  remains an ordinary, quality-preserving Base request. Nari generates queued
  phrases concurrently while this session preserves playback order.
  """

  def __init__(
    self,
    runtime: NariTTSRuntime,
    request: BatchGenerationRequest,
    *,
    seed: int | None,
    cancel_event: asyncio.Event,
  ) -> None:
    """Store fixed prompt state and initialize bounded phrase queues."""
    self._runtime = runtime
    self._request = request
    self._seed = seed
    self._cancel_event = cancel_event
    self._expected_sequence = 0
    self._append_events = 0
    self._text = ""
    self._pending_text = ""
    self._input_finished = False
    self._audio_claimed = False
    self._outer_closed = False
    self._request_id = f"voice-open-live-{uuid.uuid4().hex}"
    self._outputs: asyncio.Queue[_SegmentOutput | None] = asyncio.Queue()
    self._tasks: set[asyncio.Task[None]] = set()
    self._started = asyncio.Event()

  @property
  def text(self) -> str:
    """Return every accepted text fragment in client order."""
    return self._text

  async def prepare(self) -> None:
    """Retain a uniform async construction hook for backend implementations."""

  def _consume_sequence(self, sequence: int) -> None:
    """Reject missing, duplicated, and reordered client fragments."""
    if sequence != self._expected_sequence:
      raise ValidationError(
        f"Expected live text sequence {self._expected_sequence}, got {sequence}.",
        code="invalid_sequence",
      )
    self._expected_sequence += 1

  def _boundary(self, *, final: bool) -> int:
    """Choose the next stable phrase boundary without splitting a word."""
    text = self._pending_text
    if not text:
      return 0
    sentence_ends = [match.end() for match in _SENTENCE_BOUNDARY.finditer(text)]
    for end in sentence_ends:
      if end >= _MIN_SEGMENT_CHARACTERS:
        return min(end, _MAX_SEGMENT_CHARACTERS)
    if len(text) >= _TARGET_SEGMENT_CHARACTERS:
      ceiling = min(len(text), _MAX_SEGMENT_CHARACTERS)
      boundary = max(text.rfind(" ", 0, ceiling), text.rfind("\n", 0, ceiling))
      return boundary + 1 if boundary >= _MIN_SEGMENT_CHARACTERS else ceiling
    if not final and len(text) >= _MIN_SEGMENT_CHARACTERS and text[-1].isspace():
      return len(text)
    return len(text) if final else 0

  async def _pump_segment(
    self,
    text: str,
    queue: asyncio.Queue[StreamAudioData | _SegmentFailure | None],
  ) -> None:
    """Generate one phrase immediately so later phrases can prefetch."""
    request = BatchGenerationRequest(
      text=text,
      prompt=self._request.prompt,
      language=self._request.language,
      instructions=self._request.instructions,
    )
    try:
      async for chunk in self._runtime.stream(
        request,
        seed=self._seed,
        cancel_event=self._cancel_event,
      ):
        await queue.put(chunk)
    except BaseException as error:
      await queue.put(_SegmentFailure(error))
    finally:
      await queue.put(None)

  def _queue_segment(self, text: str) -> None:
    """Start one bounded phrase prefetch and publish its ordered output slot."""
    spoken = text.strip()
    if not spoken:
      return
    queue: asyncio.Queue[StreamAudioData | _SegmentFailure | None] = asyncio.Queue(8)
    task = asyncio.create_task(
      self._pump_segment(spoken, queue),
      name="voice-nari-live-segment",
    )
    self._tasks.add(task)
    task.add_done_callback(self._tasks.discard)
    self._outputs.put_nowait(_SegmentOutput(queue, task))
    self._started.set()

  def _drain_stable_text(self, *, final: bool) -> None:
    """Move every currently stable phrase into concurrent Nari prefetch."""
    while boundary := self._boundary(final=final):
      segment = self._pending_text[:boundary]
      self._pending_text = self._pending_text[boundary:]
      self._queue_segment(segment)
      if not final:
        return

  async def append_text(self, text: str, *, sequence: int) -> str | None:
    """Append a fragment and prefetch any newly stable Base phrase."""
    if self._input_finished or self._cancel_event.is_set():
      raise ValidationError("Live text input is closed.", code="live_input_closed")
    if not text:
      raise ValidationError("Live text append cannot be empty.")
    self._consume_sequence(sequence)
    self._append_events += 1
    live_config = self._runtime._engine.config.live_input
    if self._append_events > live_config.max_input_append_events:
      raise ValidationError("Too many live text append events.", code="too_many_input_events")
    self._text += text
    self._pending_text += text
    if len(self._text) > live_config.max_pending_text_characters:
      raise ValidationError("Live text character capacity exceeded.", code="input_too_large")
    self._drain_stable_text(final=False)
    return self._request_id if self._started.is_set() else None

  async def end_text(self, *, sequence: int) -> str:
    """Flush the last partial phrase and close the ordered output queue."""
    if self._input_finished:
      raise ValidationError("Live text input is already closed.", code="live_input_closed")
    self._consume_sequence(sequence)
    self._input_finished = True
    self._drain_stable_text(final=True)
    if not self._text.strip():
      raise ValidationError("Live speech text cannot be empty.", code="empty_input")
    self._close_outer_queue()
    return self._request_id

  def _close_outer_queue(self) -> None:
    """Publish exactly one terminal marker for ordered phrase discovery."""
    if not self._outer_closed:
      self._outer_closed = True
      self._outputs.put_nowait(None)

  async def wait_started(self) -> bool:
    """Wait for the first stable phrase or cancellation before any text."""
    if self._started.is_set():
      return True
    started = asyncio.create_task(self._started.wait())
    cancelled = asyncio.create_task(self._cancel_event.wait())
    done, pending = await asyncio.wait(
      (started, cancelled),
      return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
      task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return started in done and not self._cancel_event.is_set()

  def audio(self) -> AsyncIterator[StreamAudioData]:
    """Return the single-use ordered native PCM iterator."""
    if self._audio_claimed:
      raise RuntimeError("Live audio can only be consumed once.")
    self._audio_claimed = True
    return self._audio()

  async def _audio(self) -> AsyncIterator[StreamAudioData]:
    """Drain prefetched phrase streams in original text order."""
    while not self._cancel_event.is_set():
      output = await self._outputs.get()
      if output is None:
        return
      while not self._cancel_event.is_set():
        chunk = await output.queue.get()
        if isinstance(chunk, _SegmentFailure):
          raise chunk.error
        if chunk is None:
          break
        yield chunk

  async def cancel(self) -> None:
    """Cancel every prefetched Nari request and wake protocol waiters."""
    self._cancel_event.set()
    self._started.set()
    self._close_outer_queue()
    if self._tasks:
      await asyncio.gather(*tuple(self._tasks), return_exceptions=True)


__all__ = ["NariLiveTextSession"]
