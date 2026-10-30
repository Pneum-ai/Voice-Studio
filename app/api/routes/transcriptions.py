"""Model-independent transcription upload route."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, File, Form, UploadFile

from app.api.dependencies import ServiceDependency
from app.api.schemas.common import ProblemResponse
from app.api.schemas.speech import TranscriptionResponse
from app.core.errors import ValidationError
from app.utils.uploads import temporary_upload

router = APIRouter(prefix="/v1", tags=["transcription"])


@router.post(
  "/transcriptions",
  response_model=TranscriptionResponse,
  responses={
    400: {"model": ProblemResponse},
    409: {"model": ProblemResponse},
  },
  summary="Transcribe audio",
  description="Transcribe uploaded audio with an explicitly loaded recognition model.",
)
async def transcribe_audio(
  audio: Annotated[UploadFile, File(description="Audio to transcribe")],
  model: Annotated[str, Form(min_length=1)],
  services: ServiceDependency,
  language: Annotated[str | None, Form(max_length=32)] = None,
  context: Annotated[str | None, Form(max_length=1000)] = None,
  initial_prompt: Annotated[str | None, Form(max_length=1000)] = None,
  trim_start_seconds: Annotated[
    float | None,
    Form(ge=0, description="Optional inclusive trim start in seconds"),
  ] = None,
  trim_end_seconds: Annotated[
    float | None,
    Form(gt=0, description="Optional exclusive trim end in seconds"),
  ] = None,
) -> TranscriptionResponse:
  """Store and optionally trim an upload before local transcription."""
  if context and initial_prompt:
    raise ValidationError("Send context or initial_prompt, not both.")
  async with temporary_upload(
    audio,
    max_bytes=services.settings.max_upload_bytes,
  ) as path:
    result, duration = await services.transcriptions.transcribe(
      audio_path=path,
      model=model,
      language=language,
      initial_prompt=context or initial_prompt,
      trim_start_seconds=trim_start_seconds,
      trim_end_seconds=trim_end_seconds,
    )
  return TranscriptionResponse(
    text=result.text,
    duration_seconds=duration,
    language=result.language,
    requested_language=language,
    model=model,
  )
