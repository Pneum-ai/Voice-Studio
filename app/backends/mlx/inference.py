"""GPU-owned request routing and PCM production for the unified MLX engine."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.backends.mlx.session import create_session
from app.utils.audio import AudioData
from app.utils.silence import LeadingSilenceTrimmer


@dataclass(frozen=True, slots=True)
class Request:
  """Carry validated immutable conditioning across the GPU-worker boundary."""

  identity: int
  text: str
  reference: str | None
  transcript: str | None
  language: str
  instructions: str | None
  seed: int | None
  speaker: str | None = None


@dataclass(slots=True)
class Event:
  """Return evaluated host audio, completion, or an explicit request failure."""

  identity: int
  audio: AudioData | None = None
  done: bool = False
  error: Exception | None = None
  started: bool = False


class Inference:
  """Advance one continuous session; isolate seeded work with a FIFO admission barrier."""

  def __init__(self, model: Any, model_id: str) -> None:
    """Keep all mutable GPU state on the caller's reserved MLX worker."""
    self.model = model
    self.cache_speaker = model_id == "qwen-tts-0.6B"
    self.waiting: deque[Request] = deque()
    self.session: Any = None
    self.seeded = False
    self.current: set[int] = set()
    self.trimmers: dict[int, LeadingSilenceTrimmer] = {}

  def tick(
    self,
    incoming: list[Request],
    cancelled: set[int],
    feedback: list[tuple[int, float, float]],
  ) -> list[Event]:
    """Cancel abandoned rows, admit compatible work, and advance one bounded token step."""
    self.waiting.extend(incoming)
    events = []
    for identity in cancelled:
      self.waiting = deque(item for item in self.waiting if item.identity != identity)
      if self.session is not None and identity in self.current:
        self.session.cancel(identity)
      self.current.discard(identity)
      self.trimmers.pop(identity, None)
      events.append(Event(identity, done=True))
    if self.session is not None:
      for values in feedback:
        self.session.playback_received(*values)
      if self.session.idle:
        self.session.close()
        self.session = None
    try:
      self._admit()
      if self.session is None:
        return events
      native = self.session.step()
      events.extend(Event(identity, started=True) for identity in self.session.admitted)
      self.session.admitted.clear()
      for item in native:
        identity = item.sequence_id
        if item.error is not None:
          events.append(Event(identity, done=True, error=item.error))
        else:
          trimmer = self.trimmers[identity]
          if item.audio is not None:
            audio = AudioData(np.asarray(item.audio, dtype=np.float32), item.sample_rate)
            playable = trimmer.process(audio)
            if playable is not None and playable.samples.size:
              events.append(Event(identity, audio=playable))
          if item.done:
            tail = trimmer.finish()
            if tail is not None and tail.samples.size:
              events.append(Event(identity, audio=tail))
            events.append(Event(identity, done=True))
        if item.done:
          self.current.discard(identity)
          self.trimmers.pop(identity, None)
    except Exception as error:
      # A failed native step can leave shared caches half-updated. Fail that session
      # explicitly and recreate it for later work rather than emitting corrupted PCM.
      failed = self.current | {item.identity for item in self.waiting}
      events.extend(Event(identity, done=True, error=error) for identity in failed)
      self.waiting.clear()
      self.current.clear()
      self.trimmers.clear()
      if self.session is not None:
        self.session.close()
        self.session = None
    return events

  def _admit(self) -> None:
    """Batch independent voices while keeping explicit seeds reproducible and FIFO-fair."""
    from mlx_audio.tts.continuous import TTSBatchItem  # pyright: ignore[reportMissingImports]

    if self.session is None and self.waiting:
      import mlx.core as mx  # pyright: ignore[reportMissingImports]

      seed = self.waiting[0].seed
      self.seeded = seed is not None
      if seed is not None:
        mx.random.seed(seed)
      self.session = create_session(
        self.model, capacity=1 if self.seeded else 8, cache_speaker=self.cache_speaker
      )
    if self.session is None or (self.seeded and self.current):
      return
    items = []
    while self.waiting:
      request = self.waiting[0]
      if request.seed is not None and not self.seeded:
        break
      self.waiting.popleft()
      self.current.add(request.identity)
      self.trimmers[request.identity] = LeadingSilenceTrimmer(
        sample_rate=24000, threshold_dbfs=-45, max_ms=3000, preroll_ms=10, min_output_ms=320
      )
      items.append(
        TTSBatchItem(
          sequence_id=request.identity,
          text=request.text,
          voice=request.speaker,
          ref_audio=request.reference,
          ref_text=request.transcript,
          instruct=request.instructions,
          extra={"language": request.language},
        )
      )
      if self.seeded:
        break
    self.session.add(items)

  def close(self) -> None:
    """Release every request and reference on the GPU-owning thread."""
    if self.session is not None:
      self.session.close()
      self.session = None
    self.waiting.clear()
    self.current.clear()
    self.trimmers.clear()
    self.model._icl_cache.clear()
