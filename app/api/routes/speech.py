"""Regular HTTP speech generation route."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter
from fastapi.responses import Response

from app.api.dependencies import ServiceDependency, SessionDependency
from app.api.schemas.common import ProblemResponse
from app.api.schemas.speech import SpeechRequest
from app.utils.audio import wav_bytes

router = APIRouter(prefix="/v1", tags=["speech"])


@router.post(
  "/speech",
  response_class=Response,
  responses={
    200: {
      "content": {"audio/wav": {}},
      "description": "Complete generated PCM WAV audio.",
    },
    400: {"model": ProblemResponse},
    404: {"model": ProblemResponse},
    409: {"model": ProblemResponse},
    429: {"model": ProblemResponse},
    503: {"model": ProblemResponse},
    504: {"model": ProblemResponse},
  },
  summary="Generate speech",
  description="Generate an unsaved WAV with a selected loaded Qwen model.",
)
async def generate_speech(
  payload: SpeechRequest,
  services: ServiceDependency,
  session: SessionDependency,
) -> Response:
  """Return a complete WAV without persisting generation metadata."""
  _, audio = await services.speech.generate(
    session,
    profile_id=payload.profile_id,
    speaker=payload.speaker,
    text=payload.text,
    language=payload.language,
    model=payload.model,
    seed=payload.seed,
    instructions=payload.instructions,
    style_controls=payload.style_controls,
  )
  encoded = await asyncio.to_thread(wav_bytes, audio)
  return Response(
    content=encoded,
    media_type="audio/wav",
    headers={
      "Cache-Control": "no-store",
      "Content-Length": str(len(encoded)),
      "X-Audio-Sample-Rate": str(audio.sample_rate),
    },
  )
