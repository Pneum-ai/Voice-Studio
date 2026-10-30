"""Persistent raw-PCM WebSocket transport for low-latency speech clients."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError as PydanticValidationError

from app.api.routes.speech_live import LiveOperation, run_live_generation
from app.api.schemas.generations import GenerationResponse
from app.api.schemas.speech import (
  StreamCancelRequest,
  StreamGenerateRequest,
  StreamLiveAppendRequest,
  StreamLiveCancelRequest,
  StreamLiveEndRequest,
  StreamLiveStartRequest,
  StreamPingRequest,
)
from app.core.errors import InferenceTimeoutError, ServiceError
from app.services.container import Services
from app.services.speech_diagnostics import collect_diagnostics
from app.utils.audio import AudioData, PCM16AudioData, StreamAudioData, pcm_s16le_bytes

router = APIRouter(tags=["speech"])
LOGGER = logging.getLogger(__name__)
SendJson = Callable[[dict[str, Any]], Awaitable[None]]
SendBytes = Callable[[bytes], Awaitable[None]]


async def _send_error(
  send_json: SendJson,
  *,
  request_id: str | None,
  status: int,
  code: str,
  title: str,
  detail: str,
  retry_after_ms: int | None = None,
) -> None:
  """Send one protocol error while preserving the connection when possible."""
  try:
    payload: dict[str, Any] = {
      "type": "error",
      "request_id": request_id,
      "status": status,
      "code": code,
      "title": title,
      "detail": detail,
    }
    if retry_after_ms is not None:
      payload["retry_after_ms"] = retry_after_ms
    await send_json(payload)
  except (RuntimeError, WebSocketDisconnect):
    return


async def _run_generation(
  services: Services,
  request: StreamGenerateRequest,
  cancel_event: asyncio.Event,
  send_json: SendJson,
  send_bytes: SendBytes,
) -> None:
  """Run one generation while the connection receiver remains responsive."""
  try:
    chunks: list[StreamAudioData] = []
    with collect_diagnostics(request.diagnostics) as diagnostics:
      async with services.database.sessions() as session:
        async with services.speech.prepared(
          session,
          profile_id=request.profile_id,
          speaker=request.speaker,
          text=request.text,
          language=request.language,
          model=request.model,
          instructions=request.instructions,
          style_controls=request.style_controls,
        ) as prepared:
          model = await services.models.get_model(request.model)
          if cancel_event.is_set():
            await send_json(
              {
                "type": "cancelled",
                "request_id": request.request_id,
              }
            )
            return
          await send_json(
            {
              "type": "started",
              "request_id": request.request_id,
              "encoding": "pcm_s16le",
              "sample_rate": 24_000,
              "channels": 1,
              "native_streaming": model.native_streaming,
            }
          )
          async for chunk in services.speech.stream_prepared(
            prepared,
            seed=request.seed,
            cancel_event=cancel_event,
          ):
            if cancel_event.is_set():
              break
            if chunk.sample_rate != 24_000:
              raise RuntimeError(
                f"Streaming backend returned unexpected sample rate {chunk.sample_rate}."
              )
            encoded = (
              chunk.pcm_s16le
              if isinstance(chunk, PCM16AudioData)
              else await asyncio.to_thread(pcm_s16le_bytes, chunk)
            )
            await send_bytes(encoded)
            if request.store:
              chunks.append(chunk)
        if cancel_event.is_set():
          await send_json(
            {
              "type": "cancelled",
              "request_id": request.request_id,
            }
          )
          return
        generation_response = None
        if request.store:
          if not chunks:
            raise RuntimeError("The backend completed without producing audio.")
          audio = AudioData(
            np.concatenate([chunk.samples for chunk in chunks]).astype(np.float32),
            chunks[0].sample_rate,
          )
          generation = await services.generations.save_generation(
            session,
            profile_id=prepared.voice.id,
            profile_name=prepared.voice.name,
            input_text=request.text,
            language=request.language,
            model=request.model,
            seed=request.seed,
            instructions=request.instructions,
            audio=audio,
          )
          generation_response = GenerationResponse.from_entity(generation).model_dump(mode="json")
    finished: dict[str, Any] = {
      "type": "finished",
      "request_id": request.request_id,
      "generation": generation_response,
    }
    if diagnostics is not None:
      finished["diagnostics"] = diagnostics
    await send_json(finished)
  except ServiceError as error:
    await _send_error(
      send_json,
      request_id=request.request_id,
      status=error.status,
      code=error.code,
      title=error.title,
      detail=error.detail,
      retry_after_ms=error.retry_after_ms,
    )
  except (RuntimeError, TypeError, ValueError) as error:
    await _send_error(
      send_json,
      request_id=request.request_id,
      status=500,
      code="generation_failed",
      title="Speech generation failed",
      detail=str(error),
    )
  except Exception as error:
    LOGGER.exception("Unexpected streaming generation failure", exc_info=error)
    await _send_error(
      send_json,
      request_id=request.request_id,
      status=500,
      code="internal_error",
      title="Speech generation failed",
      detail="The generation could not be completed.",
    )


def _parse_message(payload: object) -> object:
  """Validate one client JSON message by its protocol type discriminator."""
  if not isinstance(payload, dict):
    raise ValueError("WebSocket messages must be JSON objects.")
  message_type = payload.get("type")
  if message_type == "generate":
    return StreamGenerateRequest.model_validate(payload)
  if message_type == "cancel":
    return StreamCancelRequest.model_validate(payload)
  if message_type == "ping":
    return StreamPingRequest.model_validate(payload)
  if message_type == "request.start":
    return StreamLiveStartRequest.model_validate(payload)
  if message_type == "input_text.append":
    return StreamLiveAppendRequest.model_validate(payload)
  if message_type == "input_text.end":
    return StreamLiveEndRequest.model_validate(payload)
  if message_type == "response.cancel":
    return StreamLiveCancelRequest.model_validate(payload)
  raise ValueError("Unknown WebSocket message type.")


@router.websocket("/v1/speech/stream")
async def stream_speech(websocket: WebSocket) -> None:
  """Serve sequential speech operations while accepting pings and cancellation."""
  await websocket.accept()
  services: Services = websocket.app.state.services
  send_lock = asyncio.Lock()
  active_task: asyncio.Task[None] | None = None
  active_request: StreamGenerateRequest | StreamLiveStartRequest | None = None
  live_operation: LiveOperation | None = None
  cancel_event: asyncio.Event | None = None

  async def send_json(payload: dict[str, Any]) -> None:
    """Serialize JSON writes with binary audio frames for frame ordering."""
    async with send_lock:
      await websocket.send_json(payload)

  async def send_bytes(payload: bytes) -> None:
    """Apply WebSocket backpressure by awaiting each binary frame write."""
    async with send_lock:
      try:
        async with asyncio.timeout(services.settings.stream_send_timeout_seconds):
          await websocket.send_bytes(payload)
      except TimeoutError as error:
        raise InferenceTimeoutError(
          "The speech client did not accept streamed audio before its deadline."
        ) from error

  try:
    while True:
      message = await websocket.receive()
      if message["type"] == "websocket.disconnect":
        break
      if message.get("text") is None:
        await _send_error(
          send_json,
          request_id=None,
          status=400,
          code="invalid_message",
          title="Invalid WebSocket message",
          detail="Client control messages must be JSON text frames.",
        )
        continue
      try:
        parsed = _parse_message(json.loads(message["text"]))
      except (json.JSONDecodeError, PydanticValidationError, ValueError) as error:
        await _send_error(
          send_json,
          request_id=None,
          status=400,
          code="invalid_message",
          title="Invalid WebSocket message",
          detail=str(error),
        )
        continue
      if isinstance(parsed, StreamPingRequest):
        await send_json({"type": "pong"})
        continue
      if isinstance(parsed, StreamCancelRequest):
        if (
          active_task is not None
          and not active_task.done()
          and active_request is not None
          and parsed.request_id == active_request.request_id
          and cancel_event is not None
        ):
          await send_json(
            {
              "type": "cancel_requested",
              "request_id": parsed.request_id,
            }
          )
          cancel_event.set()
          if live_operation is not None:
            await live_operation.cancel()
        else:
          await _send_error(
            send_json,
            request_id=parsed.request_id,
            status=404,
            code="request_not_active",
            title="Request is not active",
            detail="No active generation matches that request ID.",
          )
        continue
      if isinstance(parsed, StreamLiveCancelRequest):
        if active_task is not None and not active_task.done() and live_operation is not None:
          await send_json(
            {
              "type": "cancel_requested",
              "request_id": live_operation.request.request_id,
            }
          )
          await live_operation.cancel()
        else:
          await _send_error(
            send_json,
            request_id=None,
            status=404,
            code="request_not_active",
            title="Request is not active",
            detail="No active live generation is configured on this connection.",
          )
        continue
      if isinstance(parsed, (StreamLiveAppendRequest, StreamLiveEndRequest)):
        if active_task is None or active_task.done() or live_operation is None:
          await _send_error(
            send_json,
            request_id=None,
            status=409,
            code="live_request_not_configured",
            title="Live request is not configured",
            detail="Send request.start and wait for request.configured before text input.",
          )
          continue
        try:
          stream = await live_operation.wait_stream()
          if isinstance(parsed, StreamLiveAppendRequest):
            engine_request_id = await stream.append_text(
              parsed.text,
              sequence=parsed.sequence,
            )
            event = "input_text.append"
          else:
            engine_request_id = await stream.end_text(sequence=parsed.sequence)
            event = "input_text.end"
          await send_json(
            {
              "type": "input_text.ack",
              "event": event,
              "sequence": parsed.sequence,
              "request_id": live_operation.request.request_id,
              "engine_request_id": engine_request_id,
            }
          )
        except ServiceError as error:
          await _send_error(
            send_json,
            request_id=live_operation.request.request_id,
            status=error.status,
            code=error.code,
            title=error.title,
            detail=error.detail,
            retry_after_ms=error.retry_after_ms,
          )
          await live_operation.cancel()
        except (RuntimeError, TypeError, ValueError) as error:
          await _send_error(
            send_json,
            request_id=live_operation.request.request_id,
            status=400,
            code="invalid_live_input",
            title="Invalid live speech input",
            detail=str(error),
          )
          await live_operation.cancel()
        continue
      assert isinstance(parsed, (StreamGenerateRequest, StreamLiveStartRequest))
      if active_task is not None and not active_task.done():
        await _send_error(
          send_json,
          request_id=parsed.request_id,
          status=409,
          code="connection_busy",
          title="Connection is busy",
          detail="Wait for the active request to finish or cancel it.",
        )
        continue
      if active_task is not None:
        await active_task
      active_request = parsed
      if isinstance(parsed, StreamLiveStartRequest):
        live_operation = LiveOperation(parsed)
        cancel_event = live_operation.cancel_event
        active_task = asyncio.create_task(
          run_live_generation(
            services,
            live_operation,
            send_json,
            send_bytes,
            _send_error,
          )
        )
      else:
        live_operation = None
        cancel_event = asyncio.Event()
        active_task = asyncio.create_task(
          _run_generation(
            services,
            parsed,
            cancel_event,
            send_json,
            send_bytes,
          )
        )
  except WebSocketDisconnect:
    pass
  finally:
    if cancel_event is not None:
      cancel_event.set()
    if live_operation is not None:
      with suppress(Exception):
        await live_operation.cancel()
    if active_task is not None:
      if not active_task.done():
        active_task.cancel()
      with suppress(asyncio.CancelledError):
        await active_task
