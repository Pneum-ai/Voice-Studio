"""Nari-style incremental text operation for the shared speech WebSocket."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from app.api.schemas.generations import GenerationResponse
from app.api.schemas.speech import StreamLiveStartRequest
from app.backends.protocols import LiveTextStream
from app.core.errors import ServiceError
from app.services.container import Services
from app.services.speech_diagnostics import collect_diagnostics
from app.utils.audio import AudioData, PCM16AudioData, StreamAudioData, pcm_s16le_bytes

LOGGER = logging.getLogger(__name__)
SendJson = Callable[[dict[str, Any]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]
SendError = Callable[..., Awaitable[None]]


@dataclass(slots=True)
class LiveOperation:
  """Share one live backend session between receiver and audio sender tasks."""

  request: StreamLiveStartRequest
  cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
  stream: LiveTextStream | None = None
  setup_error: BaseException | None = None
  ready: asyncio.Event = field(default_factory=asyncio.Event)

  def attach(self, stream: LiveTextStream) -> None:
    """Publish the configured backend stream to the socket receiver."""
    self.stream = stream
    self.ready.set()

  def fail_setup(self, error: BaseException) -> None:
    """Wake receiver calls when profile or backend setup failed."""
    self.setup_error = error
    self.ready.set()

  async def wait_stream(self) -> LiveTextStream:
    """Wait until the runner resolves fixed profile and model state."""
    await self.ready.wait()
    if self.setup_error is not None:
      raise RuntimeError("Live speech setup failed.") from self.setup_error
    if self.stream is None:
      raise RuntimeError("Live speech setup completed without a backend stream.")
    return self.stream

  async def cancel(self) -> None:
    """Cancel before or after the backend engine request is submitted."""
    self.cancel_event.set()
    if self.stream is not None:
      await self.stream.cancel()


async def _encoded_chunk(chunk: StreamAudioData) -> bytes:
  """Return transport PCM without converting direct Nari byte chunks."""
  if chunk.sample_rate != 24_000:
    raise RuntimeError(f"Streaming backend returned unexpected sample rate {chunk.sample_rate}.")
  if isinstance(chunk, PCM16AudioData):
    return chunk.pcm_s16le
  return await asyncio.to_thread(pcm_s16le_bytes, chunk)


async def _save_generation(
  services: Services,
  session: Any,
  operation: LiveOperation,
  prepared: Any,
  chunks: list[StreamAudioData],
) -> dict[str, Any]:
  """Persist an opted-in live request after its final PCM chunk arrives."""
  if not chunks:
    raise RuntimeError("The backend completed without producing audio.")
  audio = AudioData(
    np.concatenate([chunk.samples for chunk in chunks]).astype(np.float32),
    chunks[0].sample_rate,
  )
  assert operation.stream is not None
  request = operation.request
  generation = await services.generations.save_generation(
    session,
    profile_id=prepared.voice.id,
    profile_name=prepared.voice.name,
    input_text=operation.stream.text,
    language=request.language,
    model=request.model,
    seed=request.seed,
    instructions=request.instructions,
    audio=audio,
  )
  return GenerationResponse.from_entity(generation).model_dump(mode="json")


async def run_live_generation(
  services: Services,
  operation: LiveOperation,
  send_json: SendJson,
  send_bytes: SendBytes,
  send_error: SendError,
) -> None:
  """Resolve one live request and forward PCM without blocking text appends."""
  request = operation.request
  try:
    chunks: list[StreamAudioData] = []
    generation_response = None
    with collect_diagnostics(request.diagnostics) as diagnostics:
      async with services.database.sessions() as session:
        async with services.speech.prepared_live(
          session,
          profile_id=request.profile_id,
          speaker=request.speaker,
          language=request.language,
          model=request.model,
          instructions=request.instructions,
          style_controls=request.style_controls,
        ) as prepared:
          model = await services.models.get_model(request.model)
          async with services.speech.batcher.streaming_section(
            prepared.model,
          ):
            stream = await services.speech.open_live_text(
              prepared,
              seed=request.seed,
              cancel_event=operation.cancel_event,
            )
            operation.attach(stream)
            await send_json(
              {
                "type": "request.configured",
                "request_id": request.request_id,
                "live_text": True,
              }
            )
            if not await stream.wait_started():
              await send_json({"type": "cancelled", "request_id": request.request_id})
              return
            await send_json(
              {
                "type": "started",
                "request_id": request.request_id,
                "encoding": "pcm_s16le",
                "sample_rate": 24_000,
                "channels": 1,
                "native_streaming": model.native_streaming,
                "live_text": True,
              }
            )
            async for chunk in stream.audio():
              if operation.cancel_event.is_set():
                break
              await send_bytes(await _encoded_chunk(chunk))
              if request.store:
                chunks.append(chunk)
        if operation.cancel_event.is_set():
          await send_json({"type": "cancelled", "request_id": request.request_id})
          return
        if request.store:
          generation_response = await _save_generation(
            services,
            session,
            operation,
            prepared,
            chunks,
          )
    finished: dict[str, Any] = {
      "type": "finished",
      "request_id": request.request_id,
      "generation": generation_response,
      "live_text": True,
    }
    if diagnostics is not None:
      finished["diagnostics"] = diagnostics
    await send_json(finished)
  except ServiceError as error:
    operation.fail_setup(error)
    await send_error(
      send_json,
      request_id=request.request_id,
      status=error.status,
      code=error.code,
      title=error.title,
      detail=error.detail,
      retry_after_ms=error.retry_after_ms,
    )
  except (RuntimeError, TypeError, ValueError) as error:
    operation.fail_setup(error)
    await send_error(
      send_json,
      request_id=request.request_id,
      status=500,
      code="generation_failed",
      title="Speech generation failed",
      detail=str(error),
    )
  except asyncio.CancelledError:
    with suppress(Exception):
      await operation.cancel()
    raise
  except Exception as error:
    operation.fail_setup(error)
    LOGGER.exception("Unexpected live streaming generation failure", exc_info=error)
    await send_error(
      send_json,
      request_id=request.request_id,
      status=500,
      code="internal_error",
      title="Speech generation failed",
      detail="The live generation could not be completed.",
    )


__all__ = ["LiveOperation", "run_live_generation"]
