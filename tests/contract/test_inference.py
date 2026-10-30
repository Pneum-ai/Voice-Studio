"""Speech, transcription, model, error, and WebSocket contracts."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.errors import OverloadedError
from tests.helpers import reference_wav_bytes


def test_health_models_http_speech_and_transcription(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Exercise core direct-use endpoints without network or hardware."""
  health = client.get("/health")
  assert health.status_code == 200
  assert health.json()["status"] == "ready"
  models = client.get("/v1/models").json()["items"]
  qwen = next(item for item in models if item["id"] == "qwen-tts-1.7B")
  assert qwen["native_streaming"] is True
  assert qwen["load_state"] == "loaded"
  turbo = next(item for item in models if item["id"] == "whisper-turbo")
  assert turbo["load_state"] == "loaded"

  speech = client.post(
    "/v1/speech",
    json={
      "profile_id": profile_with_sample["id"],
      "text": "Return this as a regular WAV.",
      "model": "qwen-tts-1.7B",
    },
  )
  assert speech.status_code == 200, speech.text
  assert speech.headers["content-type"].startswith("audio/wav")
  assert int(speech.headers["content-length"]) == len(speech.content)

  transcription = client.post(
    "/v1/transcriptions",
    files={"audio": ("speech.wav", reference_wav_bytes())},
    data={
      "model": "whisper-turbo",
      "language": "en",
      "trim_start_seconds": "0.5",
      "trim_end_seconds": "1.5",
    },
  )
  assert transcription.status_code == 200, transcription.text
  assert transcription.json()["text"] == "deterministic transcript"
  assert transcription.json()["duration_seconds"] == 1.0


def test_openai_audio_endpoints_use_standard_fields(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Accept OpenAI client payloads while retaining local profile and model IDs."""
  speech = client.post(
    "/v1/audio/speech",
    json={
      "model": "qwen-tts-1.7B",
      "input": "Stream this as raw PCM.",
      "voice": {"id": profile_with_sample["id"]},
      "response_format": "pcm",
    },
  )
  assert speech.status_code == 200, speech.text
  assert speech.headers["content-type"].startswith("audio/pcm")
  assert speech.headers["x-audio-sample-rate"] == "24000"
  assert len(speech.content) == 4800

  default_speech = client.post(
    "/v1/audio/speech",
    json={
      "model": "qwen-tts-1.7B",
      "input": "Encode the default OpenAI response format.",
      "voice": profile_with_sample["id"],
    },
  )
  assert default_speech.status_code == 200, default_speech.text
  assert default_speech.headers["content-type"].startswith("audio/mpeg")
  assert default_speech.content

  transcription = client.post(
    "/v1/audio/transcriptions",
    files={"file": ("speech.wav", reference_wav_bytes(), "audio/wav")},
    data={
      "model": "whisper-turbo",
      "language": "en",
      "prompt": "Pneum.ai",
      "response_format": "json",
    },
  )
  assert transcription.status_code == 200, transcription.text
  assert transcription.json() == {"text": "deterministic transcript"}


def test_openai_transcription_text_and_verbose_formats(client: TestClient) -> None:
  """Render the common OpenAI transcription response variants."""
  request = {
    "files": {"file": ("speech.wav", reference_wav_bytes(), "audio/wav")},
    "data": {"model": "whisper-turbo", "response_format": "text"},
  }
  text_response = client.post("/v1/audio/transcriptions", **request)
  assert text_response.status_code == 200
  assert text_response.text == "deterministic transcript"

  request["data"]["response_format"] = "verbose_json"
  verbose_response = client.post("/v1/audio/transcriptions", **request)
  assert verbose_response.status_code == 200
  assert verbose_response.json()["task"] == "transcribe"
  assert verbose_response.json()["duration"] == 2.1


def test_openai_speech_rejects_unsupported_speed(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Reject speed changes instead of silently violating the public request."""
  response = client.post(
    "/v1/audio/speech",
    json={
      "model": "qwen-tts-1.7B",
      "input": "Do not time-stretch this.",
      "voice": profile_with_sample["id"],
      "speed": 1.5,
      "response_format": "wav",
    },
  )
  assert response.status_code == 400
  assert response.json()["code"] == "unsupported_speech_speed"


def test_health_reports_failed_nari_readiness(
  client: TestClient,
) -> None:
  """Let routing stop sending speech while the API process remains observable."""
  application = cast(FastAPI, client.app)
  backend = application.state.services.backends.tts
  backend.runtime_status = SimpleNamespace(
    ready=False,
    reason="service_failed",
    active=0,
    queued=0,
    active_limit=4,
    active_ceiling=8,
    failure_id="failure-123",
    failure_detail="RuntimeError: graph slot failed",
  )

  health = client.get("/health")

  assert health.status_code == 503
  assert health.json()["status"] == "unavailable"
  assert health.json()["inference"]["failure_id"] == "failure-123"
  assert health.json()["inference"]["ready"] is False


def test_problem_responses_are_consistent(client: TestClient) -> None:
  """Return framework and domain failures as problem JSON with request IDs."""
  missing = client.get(
    "/v1/profiles/not-a-profile",
    headers={"X-Request-ID": "test-request-id"},
  )
  assert missing.status_code == 404
  assert missing.headers["content-type"].startswith("application/problem+json")
  assert missing.json()["request_id"] == "test-request-id"
  assert missing.json()["code"] == "not_found"

  invalid = client.post("/v1/profiles", json={"name": ""})
  assert invalid.status_code == 422
  assert invalid.json()["code"] == "validation_error"


def test_http_overload_is_typed_and_retryable(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Expose bounded admission as retryable load shedding instead of a 500."""
  application = cast(FastAPI, client.app)
  application.state.services.speech.generate = AsyncMock(side_effect=OverloadedError(200))

  response = client.post(
    "/v1/speech",
    json={
      "profile_id": profile_with_sample["id"],
      "text": "This request is intentionally shed.",
      "model": "qwen-tts-1.7B",
    },
  )

  assert response.status_code == 429
  assert response.headers["retry-after"] == "1"
  assert response.json()["code"] == "overloaded"


def test_model_jobs_and_prompt_cache_contract(
  client: TestClient,
) -> None:
  """Use one asynchronous desired-state contract and one cache-clear shape."""
  accepted = client.patch(
    "/v1/models/qwen-tts-0.6B",
    json={"state": "loaded"},
  )
  assert accepted.status_code == 202, accepted.text
  job = accepted.json()
  events = client.get(f"/v1/jobs/{job['id']}/events")
  assert events.status_code == 200
  assert "event: complete" in events.text
  model = client.get("/v1/models/qwen-tts-0.6B").json()
  assert model["load_state"] == "loaded"
  cleared = client.delete("/v1/cache/profile-prompts")
  assert cleared.status_code == 200
  assert isinstance(cleared.json()["deleted_files"], int)


def test_optional_runtime_install_job_contract(
  client: TestClient,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  """Install optional runtimes through the same retained-job contract as models."""
  service = cast(FastAPI, client.app).state.services.model_runtimes
  monkeypatch.setattr(service, "_run", AsyncMock())
  monkeypatch.setattr("app.services.model_runtimes.shutil.which", lambda _: "uv")

  accepted = client.post("/v1/model-runtimes/install-all")

  assert accepted.status_code == 202, accepted.text
  job = accepted.json()
  events = client.get(f"/v1/jobs/{job['id']}/events")
  assert events.status_code == 200
  assert "event: complete" in events.text


def test_persistent_websocket_streams_pcm_and_stores_audio(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Reuse one connection for ping and multiple raw-PCM speech operations."""
  profile_id = str(profile_with_sample["id"])
  with client.websocket_connect("/v1/speech/stream") as websocket:
    websocket.send_json({"type": "ping"})
    assert websocket.receive_json() == {"type": "pong"}
    websocket.send_json(
      {
        "type": "generate",
        "request_id": "first",
        "profile_id": profile_id,
        "text": "First streamed request.",
        "language": "en",
        "model": "qwen-tts-1.7B",
        "store": False,
      }
    )
    started = websocket.receive_json()
    assert started["type"] == "started"
    assert started["request_id"] == "first"
    assert started["encoding"] == "pcm_s16le"
    assert started["native_streaming"] is True
    assert len(websocket.receive_bytes()) == 2400
    assert len(websocket.receive_bytes()) == 2400
    assert websocket.receive_json() == {
      "type": "finished",
      "request_id": "first",
      "generation": None,
    }

    websocket.send_json(
      {
        "type": "generate",
        "request_id": "second",
        "profile_id": profile_id,
        "text": "Persist this streamed request.",
        "language": "en",
        "model": "qwen-tts-1.7B",
        "store": True,
      }
    )
    assert websocket.receive_json()["request_id"] == "second"
    websocket.receive_bytes()
    websocket.receive_bytes()
    finished = websocket.receive_json()
    assert finished["type"] == "finished"
    assert finished["generation"]["input_text"] == "Persist this streamed request."
    assert client.get(finished["generation"]["audio_url"]).status_code == 200


def test_websocket_cancellation_keeps_connection_reusable(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Acknowledge cancellation before completion and preserve the socket."""
  with client.websocket_connect("/v1/speech/stream") as websocket:
    websocket.send_json(
      {
        "type": "generate",
        "request_id": "cancel-me",
        "profile_id": profile_with_sample["id"],
        "text": "Cancel this request.",
        "language": "en",
        "model": "qwen-tts-1.7B",
      }
    )
    assert websocket.receive_json()["type"] == "started"
    websocket.send_json({"type": "cancel", "request_id": "cancel-me"})
    assert websocket.receive_json() == {
      "type": "cancel_requested",
      "request_id": "cancel-me",
    }
    assert websocket.receive_json() == {
      "type": "cancelled",
      "request_id": "cancel-me",
    }
    websocket.send_json({"type": "ping"})
    assert websocket.receive_json() == {"type": "pong"}
