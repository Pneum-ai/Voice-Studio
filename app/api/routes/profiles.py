"""Profile and reference-sample routes."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, File, Form, UploadFile, status
from fastapi.responses import FileResponse

from app.api.dependencies import ServiceDependency, SessionDependency
from app.api.schemas.common import DeleteResponse, ProblemResponse
from app.api.schemas.profiles import (
  ProfileCreateRequest,
  ProfileListResponse,
  ProfileResponse,
  ProfileUpdateRequest,
  SampleListResponse,
  SampleResponse,
  SampleUpdateRequest,
)
from app.utils.uploads import temporary_upload

router = APIRouter(prefix="/v1/profiles", tags=["profiles"])
ERRORS: dict[int | str, dict[str, Any]] = {
  400: {"model": ProblemResponse},
  404: {"model": ProblemResponse},
}


@router.get(
  "",
  response_model=ProfileListResponse,
  summary="List voice profiles",
)
async def list_profiles(
  services: ServiceDependency,
  session: SessionDependency,
) -> ProfileListResponse:
  """List profiles with aggregate sample counts."""
  profiles = await services.profiles.list_profiles(session)
  return ProfileListResponse(items=[ProfileResponse.from_entity(profile) for profile in profiles])


@router.post(
  "",
  response_model=ProfileResponse,
  status_code=status.HTTP_201_CREATED,
  responses={400: {"model": ProblemResponse}, 409: {"model": ProblemResponse}},
  summary="Create a voice profile",
)
async def create_profile(
  payload: ProfileCreateRequest,
  services: ServiceDependency,
  session: SessionDependency,
) -> ProfileResponse:
  """Create an empty profile that can receive reference samples."""
  profile = await services.profiles.create_profile(session, payload.name)
  return ProfileResponse.from_entity(profile)


@router.get(
  "/{profile_id}",
  response_model=ProfileResponse,
  responses={404: {"model": ProblemResponse}},
  summary="Get a voice profile",
)
async def get_profile(
  profile_id: str,
  services: ServiceDependency,
  session: SessionDependency,
) -> ProfileResponse:
  """Return one voice profile."""
  profile = await services.profiles.get_profile(session, profile_id)
  return ProfileResponse.from_entity(profile)


@router.patch(
  "/{profile_id}",
  response_model=ProfileResponse,
  responses={400: {"model": ProblemResponse}, 404: {"model": ProblemResponse}},
  summary="Rename a voice profile",
)
async def update_profile(
  profile_id: str,
  payload: ProfileUpdateRequest,
  services: ServiceDependency,
  session: SessionDependency,
) -> ProfileResponse:
  """Replace a profile's display name."""
  profile = await services.profiles.rename_profile(
    session,
    profile_id=profile_id,
    name=payload.name,
  )
  return ProfileResponse.from_entity(profile)


@router.delete(
  "/{profile_id}",
  response_model=DeleteResponse,
  responses={404: {"model": ProblemResponse}},
  summary="Delete a voice profile",
)
async def delete_profile(
  profile_id: str,
  services: ServiceDependency,
  session: SessionDependency,
) -> DeleteResponse:
  """Delete a profile, its samples, and derived prompt artifacts."""
  await services.profiles.delete_profile(session, profile_id)
  return DeleteResponse()


@router.get(
  "/{profile_id}/samples",
  response_model=SampleListResponse,
  responses={404: {"model": ProblemResponse}},
  summary="List profile samples",
)
async def list_samples(
  profile_id: str,
  services: ServiceDependency,
  session: SessionDependency,
) -> SampleListResponse:
  """List stored reference samples in creation order."""
  samples = await services.profiles.list_samples(session, profile_id)
  return SampleListResponse(items=[SampleResponse.from_entity(sample) for sample in samples])


@router.post(
  "/{profile_id}/samples",
  response_model=SampleResponse,
  status_code=status.HTTP_201_CREATED,
  responses=ERRORS,
  summary="Add a profile sample",
)
async def add_sample(
  profile_id: str,
  audio: Annotated[UploadFile, File(description="Reference speech audio")],
  transcript: Annotated[str, Form(min_length=1, max_length=5000)],
  services: ServiceDependency,
  session: SessionDependency,
  trim_start_seconds: Annotated[
    float | None,
    Form(ge=0, description="Optional inclusive trim start in seconds"),
  ] = None,
  trim_end_seconds: Annotated[
    float | None,
    Form(gt=0, description="Optional exclusive trim end in seconds"),
  ] = None,
  style_controls: Annotated[list[str] | None, Form()] = None,
) -> SampleResponse:
  """Store reference audio with its exact transcript and optional style metadata."""
  async with temporary_upload(
    audio,
    max_bytes=services.settings.max_upload_bytes,
  ) as path:
    sample = await services.profiles.add_sample(
      session,
      profile_id=profile_id,
      source_path=path,
      transcript=transcript,
      style_controls=tuple(style_controls or ()),
      trim_start_seconds=trim_start_seconds,
      trim_end_seconds=trim_end_seconds,
    )
  return SampleResponse.from_entity(sample)


@router.patch(
  "/{profile_id}/samples/{sample_id}",
  response_model=SampleResponse,
  responses=ERRORS,
  summary="Update a sample transcript",
)
async def update_sample(
  profile_id: str,
  sample_id: str,
  payload: SampleUpdateRequest,
  services: ServiceDependency,
  session: SessionDependency,
) -> SampleResponse:
  """Replace a sample transcript and optional style metadata."""
  sample = await services.profiles.update_sample_metadata(
    session,
    profile_id=profile_id,
    sample_id=sample_id,
    transcript=payload.transcript,
    style_controls=payload.style_controls,
  )
  return SampleResponse.from_entity(sample)


@router.delete(
  "/{profile_id}/samples/{sample_id}",
  response_model=DeleteResponse,
  responses={404: {"model": ProblemResponse}},
  summary="Delete a profile sample",
)
async def delete_sample(
  profile_id: str,
  sample_id: str,
  services: ServiceDependency,
  session: SessionDependency,
) -> DeleteResponse:
  """Delete one sample and all profile prompts derived from it."""
  await services.profiles.delete_sample(
    session,
    profile_id=profile_id,
    sample_id=sample_id,
  )
  return DeleteResponse()


@router.get(
  "/{profile_id}/samples/{sample_id}/audio",
  responses={404: {"model": ProblemResponse}},
  response_class=FileResponse,
  summary="Download stored sample audio",
)
async def get_sample_audio(
  profile_id: str,
  sample_id: str,
  services: ServiceDependency,
  session: SessionDependency,
) -> FileResponse:
  """Return a resampled mono reference as PCM WAV."""
  sample = await services.profiles.get_sample(
    session,
    profile_id=profile_id,
    sample_id=sample_id,
  )
  return FileResponse(
    services.profiles.sample_path(sample),
    media_type="audio/wav",
    filename=f"{sample.id}.wav",
  )
