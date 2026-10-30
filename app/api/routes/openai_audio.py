"""OpenAI-compatible speech synthesis and transcription endpoints."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from app.api.dependencies import ServiceDependency, SessionDependency
from app.api.schemas.common import ProblemResponse
from app.api.schemas.speech import OpenAISpeechRequest
from app.core.errors import ValidationError
from app.utils.audio import PCM16AudioData, encoded_audio_bytes, pcm_s16le_bytes
from app.utils.uploads import temporary_upload

router = APIRouter(prefix="/v1/audio", tags=["OpenAI-compatible audio"])
TranscriptionFormat = Literal["json", "text", "srt", "verbose_json", "vtt"]

SPEECH_MEDIA_TYPES = {
  "flac": "audio/flac",
  "mp3": "audio/mpeg",
  "pcm": "audio/pcm",
  "wav": "audio/wav",
}


def _caption_text(text: str, duration: float, response_format: TranscriptionFormat) -> str:
  """Render one whole-file caption when segment timestamps are unavailable."""
  end_ms = max(0, round(duration * 1000))
  hours, remainder = divmod(end_ms, 3_600_000)
  minutes, remainder = divmod(remainder, 60_000)
  seconds, milliseconds = divmod(remainder, 1000)
  if response_format == "srt":
    end = f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"
    return f"1\n00:00:00,000 --> {end}\n{text}\n"
  end = f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}"
  return f"WEBVTT\n\n00:00:00.000 --> {end}\n{text}\n"


@router.post(
  "/transcriptions",
  responses={
    200: {
      "content": {
        "application/json": {},
        "text/plain": {},
        "text/srt": {},
        "text/vtt": {},
      },
      "description": "Transcription in the requested OpenAI response format.",
    },
    400: {"model": ProblemResponse},
    409: {"model": ProblemResponse},
  },
  summary="Create an OpenAI-compatible transcription",
  description=(
    "Transcribe a multipart `file` using OpenAI field names. Local model IDs are accepted."
  ),
)
async def create_transcription(
  file: Annotated[UploadFile, File(description="Audio file to transcribe")],
  model: Annotated[str, Form(min_length=1)],
  services: ServiceDependency,
  language: Annotated[str | None, Form(max_length=32)] = None,
  prompt: Annotated[str | None, Form(max_length=1000)] = None,
  response_format: Annotated[TranscriptionFormat, Form()] = "json",
  temperature: Annotated[float, Form(ge=0, le=1)] = 0,
) -> Response:
  """Transcribe an OpenAI-style multipart upload and format its response."""
  _ = temperature
  async with temporary_upload(file, max_bytes=services.settings.max_upload_bytes) as path:
    result, duration = await services.transcriptions.transcribe(
      audio_path=path,
      model=model,
      language=language,
      initial_prompt=prompt,
      trim_start_seconds=None,
      trim_end_seconds=None,
    )
  if response_format == "text":
    return PlainTextResponse(result.text)
  if response_format in {"srt", "vtt"}:
    return Response(
      _caption_text(result.text, duration, response_format),
      media_type=f"text/{response_format}",
    )
  if response_format == "verbose_json":
    return JSONResponse(
      {
        "task": "transcribe",
        "language": result.language,
        "duration": duration,
        "text": result.text,
        "segments": [],
        "words": [],
      }
    )
  return JSONResponse({"text": result.text})


async def _pcm_stream(
  payload: OpenAISpeechRequest,
  services: ServiceDependency,
  session: SessionDependency,
) -> AsyncGenerator[bytes, None]:
  """Yield raw OpenAI-compatible PCM chunks and cancel inference on disconnect."""
  cancel_event = asyncio.Event()
  try:
    async for _, chunk in services.speech.stream(
      session,
      profile_id=payload.profile_id,
      speaker=payload.speaker,
      text=payload.input,
      language=payload.language,
      model=payload.model,
      seed=payload.seed,
      instructions=payload.instructions,
      cancel_event=cancel_event,
    ):
      if chunk.sample_rate != 24_000:
        raise RuntimeError(
          f"Speech backend returned {chunk.sample_rate} Hz; OpenAI PCM requires 24000 Hz."
        )
      if isinstance(chunk, PCM16AudioData):
        yield chunk.pcm_s16le
      else:
        yield await asyncio.to_thread(pcm_s16le_bytes, chunk)
  finally:
    cancel_event.set()


@router.post(
  "/speech",
  responses={
    200: {
      "content": {
        "audio/flac": {},
        "audio/mpeg": {},
        "audio/pcm": {},
        "audio/wav": {},
      },
      "description": "Generated audio in the requested format.",
    },
    400: {"model": ProblemResponse},
    404: {"model": ProblemResponse},
    409: {"model": ProblemResponse},
    429: {"model": ProblemResponse},
    503: {"model": ProblemResponse},
    504: {"model": ProblemResponse},
  },
  summary="Create OpenAI-compatible speech",
  description=(
    "Generate speech using OpenAI field names. `voice` is a built-in speaker for CustomVoice, "
    'or a local profile ID / {"id": ...} for Base cloning.'
  ),
)
async def create_speech(
  payload: OpenAISpeechRequest,
  services: ServiceDependency,
  session: SessionDependency,
) -> Response:
  """Generate complete encoded audio or stream raw 24 kHz PCM."""
  if payload.speed != 1.0:
    raise ValidationError(
      "This local engine currently supports only speed 1.0.",
      code="unsupported_speech_speed",
    )
  if payload.response_format == "pcm":
    iterator = _pcm_stream(payload, services, session)
    try:
      first_chunk = await anext(iterator)
    except StopAsyncIteration as error:
      raise RuntimeError("The speech backend completed without producing audio.") from error

    async def stream_with_first_chunk() -> AsyncIterator[bytes]:
      """Send the preflight chunk followed by the remaining backend stream."""
      try:
        yield first_chunk
        async for chunk in iterator:
          yield chunk
      finally:
        await iterator.aclose()

    return StreamingResponse(
      stream_with_first_chunk(),
      media_type=SPEECH_MEDIA_TYPES["pcm"],
      headers={
        "Cache-Control": "no-store",
        "X-Audio-Sample-Rate": "24000",
        "X-Audio-Channels": "1",
      },
    )
  _, audio = await services.speech.generate(
    session,
    profile_id=payload.profile_id,
    speaker=payload.speaker,
    text=payload.input,
    language=payload.language,
    model=payload.model,
    seed=payload.seed,
    instructions=payload.instructions,
  )
  encoded = await asyncio.to_thread(
    encoded_audio_bytes,
    audio,
    payload.response_format,
  )
  return Response(
    encoded,
    media_type=SPEECH_MEDIA_TYPES[payload.response_format],
    headers={"Cache-Control": "no-store", "Content-Length": str(len(encoded))},
  )
