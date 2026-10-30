"""Saved generated-speech collection and audio routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Query, status
from fastapi.responses import FileResponse

from app.api.dependencies import ServiceDependency, SessionDependency
from app.api.schemas.common import DeleteResponse, ProblemResponse
from app.api.schemas.generations import GenerationListResponse, GenerationResponse
from app.api.schemas.speech import SpeechRequest

router = APIRouter(prefix="/v1/generations", tags=["generations"])
ERRORS: dict[int | str, dict[str, Any]] = {
  400: {"model": ProblemResponse},
  404: {"model": ProblemResponse},
  409: {"model": ProblemResponse},
}


@router.get(
  "",
  response_model=GenerationListResponse,
  responses={400: {"model": ProblemResponse}},
  summary="List saved generations",
)
async def list_generations(
  services: ServiceDependency,
  session: SessionDependency,
  limit: Annotated[int, Query(ge=1, le=100)] = 25,
  cursor: str | None = None,
  profile_id: str | None = None,
  model: str | None = None,
  query: Annotated[str | None, Query(max_length=500)] = None,
) -> GenerationListResponse:
  """List saved speech using stable cursor pagination and optional filters."""
  items, next_cursor = await services.generations.list_generations(
    session,
    limit=limit,
    cursor=cursor,
    profile_id=profile_id,
    model=model,
    query=query,
  )
  return GenerationListResponse(
    items=[GenerationResponse.from_entity(item) for item in items],
    next_cursor=next_cursor,
  )


@router.post(
  "",
  response_model=GenerationResponse,
  status_code=status.HTTP_201_CREATED,
  responses=ERRORS,
  summary="Generate and save speech",
)
async def create_generation(
  payload: SpeechRequest,
  services: ServiceDependency,
  session: SessionDependency,
) -> GenerationResponse:
  """Generate WAV audio and persist it with the caller's exact input text."""
  voice, audio = await services.speech.generate(
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
  generation = await services.generations.save_generation(
    session,
    profile_id=voice.id,
    profile_name=voice.name,
    input_text=payload.text,
    language=payload.language,
    model=payload.model,
    seed=payload.seed,
    instructions=payload.instructions,
    audio=audio,
  )
  return GenerationResponse.from_entity(generation)


@router.get(
  "/{generation_id}",
  response_model=GenerationResponse,
  responses={404: {"model": ProblemResponse}},
  summary="Get a saved generation",
)
async def get_generation(
  generation_id: str,
  services: ServiceDependency,
  session: SessionDependency,
) -> GenerationResponse:
  """Return one saved generation's complete public metadata."""
  generation = await services.generations.get_generation(session, generation_id)
  return GenerationResponse.from_entity(generation)


@router.delete(
  "/{generation_id}",
  response_model=DeleteResponse,
  responses={404: {"model": ProblemResponse}},
  summary="Delete a saved generation",
)
async def delete_generation(
  generation_id: str,
  services: ServiceDependency,
  session: SessionDependency,
) -> DeleteResponse:
  """Delete one generation and its stored WAV file."""
  await services.generations.delete_generation(session, generation_id)
  return DeleteResponse()


@router.get(
  "/{generation_id}/audio",
  response_class=FileResponse,
  responses={404: {"model": ProblemResponse}},
  summary="Download saved generated audio",
)
async def get_generation_audio(
  generation_id: str,
  services: ServiceDependency,
  session: SessionDependency,
) -> FileResponse:
  """Return one saved generation as PCM WAV."""
  generation = await services.generations.get_generation(session, generation_id)
  return FileResponse(
    services.generations.audio_path(generation),
    media_type="audio/wav",
    filename=f"{generation.id}.wav",
  )
