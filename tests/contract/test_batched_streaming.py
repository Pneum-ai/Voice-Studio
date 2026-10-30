"""Concurrent WebSocket speech requests share a pass and stay correlated."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from tests.helpers import reference_wav_bytes

REQUEST_TEXTS = {
  "alpha": "The first concurrent request.",
  "bravo": "The second concurrent request.",
  "charlie": "The third concurrent request.",
  "delta": "The fourth concurrent request.",
}


def create_profile(client: TestClient) -> str:
  """Create one profile holding a valid normalized reference sample."""
  profile = client.post("/v1/profiles", json={"name": "Batched voice"}).json()
  response = client.post(
    f"/v1/profiles/{profile['id']}/samples",
    files={"audio": ("reference.wav", reference_wav_bytes())},
    data={"transcript": "This is the exact reference transcript."},
  )
  assert response.status_code == 201, response.text
  return str(profile["id"])


def run_request(
  client: TestClient,
  profile_id: str,
  request_id: str,
) -> tuple[str, int]:
  """Run one streamed generation and return its ID and total PCM bytes."""
  with client.websocket_connect("/v1/speech/stream") as websocket:
    websocket.send_json(
      {
        "type": "generate",
        "request_id": request_id,
        "profile_id": profile_id,
        "text": REQUEST_TEXTS[request_id],
        "language": "en",
        "model": "qwen-tts-1.7B",
        "store": False,
      }
    )
    started = websocket.receive_json()
    assert started["type"] == "started"
    assert started["request_id"] == request_id
    assert started["native_streaming"] is False
    audio = websocket.receive_bytes()
    finished = websocket.receive_json()
    assert finished["type"] == "finished", finished
    assert finished["request_id"] == request_id
    return request_id, len(audio)


def test_concurrent_requests_share_a_batch_and_stay_correlated(
  batching_client: TestClient,
) -> None:
  """Prove overlapping requests batch together without crossing their audio."""
  profile_id = create_profile(batching_client)
  request_ids = list(REQUEST_TEXTS)
  with ThreadPoolExecutor(max_workers=len(request_ids)) as pool:
    results = list(
      pool.map(
        lambda request_id: run_request(batching_client, profile_id, request_id),
        request_ids,
      )
    )
  assert sorted(request_id for request_id, _ in results) == sorted(request_ids)
  assert all(received > 0 for _, received in results)
  backend = batching_client.app.state.services.backends.tts  # type: ignore[union-attr]
  assert max(backend.batch_sizes) > 1, backend.batch_sizes
