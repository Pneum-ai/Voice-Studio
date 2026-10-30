"""Speech generation, streaming, and transcription API schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.speech import CUSTOM_VOICE_MODEL


class OpenAIVoiceReference(BaseModel):
  """Identify a local cloned voice using OpenAI's custom-voice shape."""

  id: str = Field(min_length=1)


class OpenAISpeechRequest(BaseModel):
  """Accept the OpenAI speech request shape with local model and voice IDs."""

  model: str = Field(default="qwen-tts-1.7B", min_length=1)
  input: str = Field(min_length=1, max_length=4096)
  voice: str | OpenAIVoiceReference
  instructions: str | None = Field(default=None, max_length=500)
  response_format: Literal["mp3", "flac", "wav", "pcm"] = "mp3"
  speed: float = Field(default=1.0, ge=0.25, le=4.0)
  stream_format: Literal["audio"] = "audio"
  language: str = Field(default="en", min_length=1, max_length=32)
  seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807)

  @property
  def profile_id(self) -> str | None:
    """Return the local profile ID carried by either accepted voice shape."""
    if isinstance(self.voice, OpenAIVoiceReference):
      return self.voice.id
    return None if self.model == CUSTOM_VOICE_MODEL else self.voice

  @property
  def speaker(self) -> str | None:
    """Map a CustomVoice string to a built-in speaker, never a cloned profile."""
    if self.model == CUSTOM_VOICE_MODEL and isinstance(self.voice, str):
      return self.voice
    return None


class SpeechRequest(BaseModel):
  """Select a cloned profile or built-in speaker for the chosen loaded Qwen model."""

  profile_id: str | None = Field(default=None, min_length=1)
  speaker: str | None = Field(default=None, min_length=1, max_length=64)
  text: str = Field(min_length=1, max_length=5000)
  language: str = Field(default="en", min_length=1, max_length=32)
  model: str = Field(default="qwen-tts-1.7B", min_length=1)
  seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807)
  instructions: str | None = Field(default=None, max_length=500)
  style_controls: tuple[str, ...] = Field(default=(), max_length=5)


class StreamGenerateRequest(SpeechRequest):
  """Request one WebSocket speech operation."""

  type: Literal["generate"]
  request_id: str = Field(min_length=1, max_length=128)
  store: bool = False
  diagnostics: bool = False


class StreamCancelRequest(BaseModel):
  """Cancel the matching active WebSocket operation."""

  type: Literal["cancel"]
  request_id: str = Field(min_length=1, max_length=128)


class StreamPingRequest(BaseModel):
  """Request a WebSocket liveness response."""

  type: Literal["ping"]


class StreamLiveStartRequest(BaseModel):
  """Configure one appendable text request on the existing speech socket."""

  type: Literal["request.start"]
  request_id: str = Field(min_length=1, max_length=128)
  profile_id: str | None = Field(default=None, min_length=1)
  speaker: str | None = Field(default=None, min_length=1, max_length=64)
  language: str = Field(default="en", min_length=1, max_length=32)
  model: str = Field(default="qwen-tts-1.7B", min_length=1)
  seed: int | None = Field(default=None, ge=0, le=9_223_372_036_854_775_807)
  instructions: str | None = Field(default=None, max_length=500)
  style_controls: tuple[str, ...] = Field(default=(), max_length=5)
  store: bool = False
  diagnostics: bool = False


class StreamLiveAppendRequest(BaseModel):
  """Append the next ordered fragment to an active live request."""

  type: Literal["input_text.append"]
  sequence: int = Field(ge=0)
  text: str = Field(min_length=1, max_length=65_536)


class StreamLiveEndRequest(BaseModel):
  """Close live text input after flushing the final tokenizer tail."""

  type: Literal["input_text.end"]
  sequence: int = Field(ge=0)


class StreamLiveCancelRequest(BaseModel):
  """Cancel the connection's active Nari-style live request."""

  type: Literal["response.cancel"]


class TranscriptionResponse(BaseModel):
  """Return recognized text and source metadata."""

  text: str
  duration_seconds: float
  language: str | None
  requested_language: str | None
  model: str
