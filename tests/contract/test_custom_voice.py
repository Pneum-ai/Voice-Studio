"""Preset voices work across speech transports without creating cloned profiles."""

from typing import Any, cast

import pytest
from fastapi.testclient import TestClient

MODEL = "qwen-tts-1.7B-customvoice"


def load_custom_voice(client: TestClient) -> None:
  """Select the fake runtime while keeping real API validation and persistence."""
  cast(Any, client.app).state.services.backends.tts._loaded_model = MODEL


def test_preset_speech_and_library_without_profiles(client: TestClient) -> None:
  """Save a preset's name and instructions with no invalid profile foreign key."""
  load_custom_voice(client)
  payload = {"model": MODEL, "speaker": "Ryan", "text": "Hello.", "instructions": "Speak softly."}
  assert client.get("/v1/profiles").json()["items"] == []
  response = client.post("/v1/generations", json=payload)
  assert response.status_code == 201, response.text
  saved = response.json()
  assert saved["profile_id"] is None
  assert saved["profile_name"] == "Ryan"
  assert saved["instructions"] == "Speak softly."
  assert client.get(saved["audio_url"]).content.startswith(b"RIFF")
  audio = client.post(
    "/v1/audio/speech",
    json={"model": MODEL, "voice": "ryan", "input": "Hello.", "response_format": "pcm"},
  )
  assert audio.status_code == 200, audio.text
  assert audio.content


def test_preset_websocket_stream_can_be_saved(client: TestClient) -> None:
  """Stream and persist built-in speech through the same socket as cloned voices."""
  load_custom_voice(client)
  with client.websocket_connect("/v1/speech/stream") as socket:
    socket.send_json(
      {
        "type": "generate",
        "request_id": "preset",
        "model": MODEL,
        "speaker": "serena",
        "text": "Hello there.",
        "store": True,
      }
    )
    assert socket.receive_json()["type"] == "started"
    assert socket.receive_bytes()
    assert socket.receive_bytes()
    result = socket.receive_json()
    assert result["type"] == "finished", result
    assert result["generation"]["profile_name"] == "Serena"
    assert result["generation"]["profile_id"] is None


@pytest.mark.parametrize(
  "conditioning",
  [
    {},
    {"speaker": "not-a-speaker"},
    {"profile_id": "clone"},
    {"speaker": "ryan", "profile_id": "clone"},
  ],
)
def test_custom_voice_rejects_incompatible_sources(
  client: TestClient,
  conditioning: dict[str, str],
) -> None:
  """Never silently substitute a preset for a cloned or unknown voice."""
  load_custom_voice(client)
  response = client.post("/v1/speech", json={"model": MODEL, "text": "Hello.", **conditioning})
  assert response.status_code == 400, response.text


@pytest.mark.parametrize(
  "conditioning",
  [
    {"speaker": "ryan"},
    {"instructions": "Speak softly."},
    {"style_controls": ["pace_slow"]},
  ],
)
def test_base_rejects_preset_and_instruction_inputs(
  client: TestClient,
  profile_with_sample: dict[str, object],
  conditioning: dict[str, object],
) -> None:
  """Unsupported controls must fail visibly instead of implying model support."""
  response = client.post(
    "/v1/speech",
    json={
      "profile_id": profile_with_sample["id"],
      "text": "Hello.",
      **conditioning,
    },
  )
  assert response.status_code == 400, response.text
