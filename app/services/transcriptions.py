"""Validated, model-independent transcription workflow."""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

from app.backends.factory import BackendBundle
from app.backends.protocols import TranscriptionRequest, TranscriptionResult
from app.core.errors import ValidationError
from app.domain.models import ModelKind, TranscriptionLanguageMode
from app.services.models import ModelService
from app.utils.audio import (
  audio_duration_seconds,
  load_audio_selection,
  resolve_audio_selection,
  write_wav,
)
from app.utils.files import unlink_if_exists

MAX_TRANSCRIPTION_SECONDS = 60 * 60


class TranscriptionService:
  """Transcribe uploaded audio through a selected loaded recognition model."""

  def __init__(self, backends: BackendBundle, models: ModelService) -> None:
    """Store explicit backend and model lifecycle dependencies."""
    self.backends = backends
    self.models = models

  async def transcribe(
    self,
    *,
    audio_path: Path,
    model: str,
    language: str | None,
    initial_prompt: str | None,
    trim_start_seconds: float | None = None,
    trim_end_seconds: float | None = None,
  ) -> tuple[TranscriptionResult, float]:
    """Validate, transcribe, and return text with source duration in seconds."""
    clean_language = language.strip() if language else None
    clean_prompt = initial_prompt.strip() if initial_prompt else None
    if clean_language is not None and len(clean_language) > 32:
      raise ValidationError("Language cannot exceed 32 characters.")
    if clean_prompt is not None and len(clean_prompt) > 1000:
      raise ValidationError("Recognition context cannot exceed 1000 characters.")
    definition = self.models.definition(model)
    capabilities = definition.transcription
    if capabilities is None:
      raise ValidationError(f"{model} is not a transcription model.", "wrong_model_kind")
    if capabilities.language_mode is TranscriptionLanguageMode.AUTOMATIC and clean_language:
      raise ValidationError(
        f"{definition.display_name} detects language automatically.",
        "unsupported_language_option",
      )
    if capabilities.language_mode is TranscriptionLanguageMode.REQUIRED and not clean_language:
      raise ValidationError(
        f"{definition.display_name} requires a language.",
        "language_required",
      )
    if clean_language and capabilities.languages:
      clean_language = clean_language.lower()
      if clean_language not in capabilities.languages:
        raise ValidationError(
          f"{definition.display_name} does not support language {clean_language}.",
          "unsupported_language",
        )
    if clean_prompt and not capabilities.context:
      raise ValidationError(
        f"{definition.display_name} does not support recognition context.",
        "unsupported_context",
      )
    source_duration = await asyncio.to_thread(audio_duration_seconds, audio_path)
    selection = resolve_audio_selection(
      source_duration,
      start_seconds=trim_start_seconds,
      end_seconds=trim_end_seconds,
    )
    if selection.duration_seconds > MAX_TRANSCRIPTION_SECONDS:
      raise ValidationError(
        "Transcription audio cannot exceed 60 minutes.",
        "audio_too_long",
      )
    if (
      capabilities.max_audio_seconds is not None
      and selection.duration_seconds > capabilities.max_audio_seconds
    ):
      raise ValidationError(
        f"{definition.display_name} accepts at most "
        f"{capabilities.max_audio_seconds} seconds of audio.",
        "audio_too_long_for_model",
      )
    selected_path = audio_path
    staged_path: Path | None = None
    if trim_start_seconds is not None or trim_end_seconds is not None:
      descriptor, staged_name = tempfile.mkstemp(
        prefix="pneum-transcription-",
        suffix=".wav",
      )
      os.close(descriptor)
      staged_path = Path(staged_name)
      try:
        selected_audio, _ = await asyncio.to_thread(
          load_audio_selection,
          audio_path,
          start_seconds=selection.start_seconds,
          end_seconds=selection.end_seconds,
        )
        await asyncio.to_thread(write_wav, staged_path, selected_audio)
        selected_path = staged_path
      except Exception:
        await asyncio.to_thread(unlink_if_exists, staged_path)
        raise
    try:
      async with self.models.use_loaded(model, ModelKind.TRANSCRIPTION):
        result = await self.backends.stt.transcribe(
          TranscriptionRequest(
            audio_path=selected_path,
            language=clean_language,
            context=clean_prompt,
          )
        )
    finally:
      if staged_path is not None:
        await asyncio.to_thread(unlink_if_exists, staged_path)
    return TranscriptionResult(
      text=result.text.strip(),
      language=result.language,
    ), selection.duration_seconds
