"""Validated cloned and built-in speech workflows shared by HTTP and WebSocket routes."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.backends.factory import BackendBundle
from app.backends.protocols import BatchGenerationRequest, LiveTextStream
from app.core.errors import ValidationError
from app.domain.models import ModelKind
from app.domain.speech import BuiltinVoicePrompt, SpeechVoice
from app.services.models import ModelService
from app.services.profiles import ProfileService
from app.services.prompts import PromptService
from app.services.style_controls import (
  clean_style_controls,
  prepare_style_input,
  style_instruction,
)
from app.services.tts_batcher import TTSBatcher
from app.utils.audio import AudioData, StreamAudioData


@dataclass(frozen=True, slots=True)
class PreparedSpeech:
  """Hold validated inputs and a resolved backend prompt."""

  voice: SpeechVoice
  text: str
  language: str
  instructions: str | None
  style_controls: tuple[str, ...]
  prompt: object
  model: str


def clean_speech_text(text: str) -> str:
  """Validate synthesis text while preserving meaningful internal whitespace."""
  cleaned = text.strip()
  if not cleaned:
    raise ValidationError("Speech text cannot be empty.")
  if len(cleaned) > 5000:
    raise ValidationError("Speech text cannot exceed 5000 characters.")
  return cleaned


def clean_language(language: str) -> str:
  """Validate a compact language code or the automatic language selector."""
  cleaned = language.strip()
  if not cleaned:
    raise ValidationError("Language cannot be empty.")
  if len(cleaned) > 32:
    raise ValidationError("Language cannot exceed 32 characters.")
  return cleaned


def clean_instructions(instructions: str | None) -> str | None:
  """Normalize optional voice-delivery instructions."""
  if instructions is None:
    return None
  cleaned = instructions.strip()
  if not cleaned:
    return None
  if len(cleaned) > 500:
    raise ValidationError("Instructions cannot exceed 500 characters.")
  return cleaned


class SpeechService:
  """Generate speech through the selected loaded Qwen backend."""

  def __init__(
    self,
    backends: BackendBundle,
    models: ModelService,
    profiles: ProfileService,
    prompts: PromptService,
    batcher: TTSBatcher,
  ) -> None:
    """Store explicit backend, model, profile, prompt, and batching dependencies."""
    self.backends = backends
    self.models = models
    self.profiles = profiles
    self.prompts = prompts
    self.batcher = batcher

  async def _resolve_voice(
    self,
    session: AsyncSession,
    model: str,
    profile_id: str | None,
    speaker: str | None,
    instructions: str | None,
  ) -> tuple[SpeechVoice, object]:
    """Reject incompatible conditioning before looking up profiles or running inference."""
    capabilities = self.models.definition(model).speech
    if capabilities is None:
      raise ValidationError("The selected model does not support speech generation.")
    if instructions and not capabilities.instructions:
      raise ValidationError("Base models do not support instructions or delivery style controls.")
    if capabilities.voice_mode == "builtin":
      if profile_id is not None:
        raise ValidationError("CustomVoice uses built-in speakers, not cloned voice profiles.")
      selected = (speaker or "").strip().lower()
      if selected not in capabilities.speakers:
        raise ValidationError("Choose one of the model's built-in speakers.")
      return SpeechVoice(None, selected.replace("_", " ").title()), BuiltinVoicePrompt(selected)
    if speaker is not None or not profile_id:
      raise ValidationError("Base models require a cloned voice profile, not a built-in speaker.")
    profile = await self.profiles.get_profile(session, profile_id)
    prompt = await self.prompts.get_prompt(session, profile_id=profile_id, model_id=model)
    return SpeechVoice(profile.id, profile.name), prompt

  async def _generate_prepared(
    self,
    prepared: PreparedSpeech,
    *,
    seed: int | None,
    cancel_event: asyncio.Event | None = None,
  ) -> AudioData:
    """Generate one waveform through the batcher so requests can share a pass."""
    return await self.batcher.submit(
      model_id=prepared.model,
      request=BatchGenerationRequest(
        text=prepared.text,
        prompt=prepared.prompt,
        language=prepared.language,
        instructions=prepared.instructions,
      ),
      seed=seed,
      cancel_event=cancel_event,
    )

  @asynccontextmanager
  async def prepared(
    self,
    session: AsyncSession,
    *,
    profile_id: str | None,
    speaker: str | None = None,
    text: str,
    language: str,
    model: str,
    instructions: str | None,
    style_controls: tuple[str, ...] = (),
  ) -> AsyncIterator[PreparedSpeech]:
    """Resolve a prompt and hold the model lifecycle guard through inference."""
    spoken_text, clean_controls = prepare_style_input(text, style_controls)
    clean_text = clean_speech_text(spoken_text)
    clean_lang = clean_language(language)
    clean_instruction = style_instruction(clean_instructions(instructions), clean_controls)
    async with self.models.use_loaded(model, ModelKind.TTS):
      voice, prompt = await self._resolve_voice(
        session, model, profile_id, speaker, clean_instruction
      )
      yield PreparedSpeech(
        voice=voice,
        text=clean_text,
        language=clean_lang,
        instructions=clean_instruction,
        style_controls=clean_controls,
        prompt=prompt,
        model=model,
      )

  @asynccontextmanager
  async def prepared_live(
    self,
    session: AsyncSession,
    *,
    profile_id: str | None,
    speaker: str | None = None,
    language: str,
    model: str,
    instructions: str | None,
    style_controls: tuple[str, ...] = (),
  ) -> AsyncIterator[PreparedSpeech]:
    """Resolve fixed voice state while target text arrives incrementally."""
    clean_lang = clean_language(language)
    clean_controls = clean_style_controls(style_controls)
    clean_instruction = style_instruction(clean_instructions(instructions), clean_controls)
    async with self.models.use_loaded(model, ModelKind.TTS):
      voice, prompt = await self._resolve_voice(
        session, model, profile_id, speaker, clean_instruction
      )
      yield PreparedSpeech(
        voice=voice,
        text="",
        language=clean_lang,
        instructions=clean_instruction,
        style_controls=clean_controls,
        prompt=prompt,
        model=model,
      )

  async def open_live_text(
    self,
    prepared: PreparedSpeech,
    *,
    seed: int | None,
    cancel_event: asyncio.Event,
  ) -> LiveTextStream:
    """Open one Nari live-input request after its model is fixed."""
    if not getattr(self.backends.tts, "supports_live_text", False):
      raise ValidationError(
        "Live text input requires the Nari inference engine.",
        code="live_text_not_supported",
      )
    return await self.backends.tts.open_live_text(
      prompt=prepared.prompt,
      language=prepared.language,
      seed=seed,
      instructions=prepared.instructions,
      cancel_event=cancel_event,
    )

  async def generate(
    self,
    session: AsyncSession,
    *,
    profile_id: str | None,
    speaker: str | None = None,
    text: str,
    language: str,
    model: str,
    seed: int | None,
    instructions: str | None,
    style_controls: tuple[str, ...] = (),
  ) -> tuple[SpeechVoice, AudioData]:
    """Generate a complete waveform and return its resolved voice."""
    async with self.prepared(
      session,
      profile_id=profile_id,
      speaker=speaker,
      text=text,
      language=language,
      model=model,
      instructions=instructions,
      style_controls=style_controls,
    ) as prepared:
      audio = await self._generate_prepared(prepared, seed=seed)
    return prepared.voice, audio

  async def stream(
    self,
    session: AsyncSession,
    *,
    profile_id: str | None,
    speaker: str | None = None,
    text: str,
    language: str,
    model: str,
    seed: int | None,
    instructions: str | None,
    cancel_event: asyncio.Event,
    style_controls: tuple[str, ...] = (),
  ) -> AsyncIterator[tuple[SpeechVoice, StreamAudioData]]:
    """Yield native backend chunks, or one completed chunk on PyTorch Qwen."""
    async with self.prepared(
      session,
      profile_id=profile_id,
      speaker=speaker,
      text=text,
      language=language,
      model=model,
      instructions=instructions,
      style_controls=style_controls,
    ) as prepared:
      async for chunk in self.stream_prepared(
        prepared,
        seed=seed,
        cancel_event=cancel_event,
      ):
        yield prepared.voice, chunk

  async def stream_prepared(
    self,
    prepared: PreparedSpeech,
    *,
    seed: int | None,
    cancel_event: asyncio.Event,
  ) -> AsyncIterator[StreamAudioData]:
    """Yield backend audio for inputs that were already validated and resolved.

    Native streaming backends emit incrementally and bypass batching. Complete
    waveform backends go through the batcher so concurrent requests can share
    one generation pass, then emit their single chunk.
    """
    if not self.backends.tts.supports_streaming:
      audio = await self._generate_prepared(
        prepared,
        seed=seed,
        cancel_event=cancel_event,
      )
      if not cancel_event.is_set():
        yield audio
      return
    async with self.batcher.streaming_section(prepared.model):
      async for chunk in self._stream_native(prepared, seed=seed, cancel_event=cancel_event):
        yield chunk

  async def _stream_native(
    self,
    prepared: PreparedSpeech,
    *,
    seed: int | None,
    cancel_event: asyncio.Event,
  ) -> AsyncIterator[StreamAudioData]:
    """Yield incremental chunks from a backend that generates them natively."""
    async for chunk in self.backends.tts.stream(
      text=prepared.text,
      prompt=prepared.prompt,
      language=prepared.language,
      seed=seed,
      instructions=prepared.instructions,
      cancel_event=cancel_event,
    ):
      yield chunk
