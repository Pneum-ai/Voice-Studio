"""Contract coverage for additive Nari-style live text events."""

from __future__ import annotations

import json
from typing import cast

from fastapi.testclient import TestClient


def _receive_frame(websocket: object) -> tuple[str, object]:
  """Normalize one TestClient WebSocket frame into JSON or binary data."""
  message = websocket.receive()  # type: ignore[attr-defined]
  if message.get("bytes") is not None:
    return "bytes", message["bytes"]
  return "json", json.loads(message["text"])


def test_live_text_emits_pcm_before_input_end_and_stores_full_text(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Append text during synthesis without changing the legacy stream contract."""
  with client.websocket_connect("/v1/speech/stream") as websocket:
    websocket.send_json(
      {
        "type": "request.start",
        "request_id": "live-one",
        "profile_id": profile_with_sample["id"],
        "language": "en",
        "model": "qwen-tts-1.7B",
        "store": True,
      }
    )
    configured = websocket.receive_json()
    assert configured == {
      "type": "request.configured",
      "request_id": "live-one",
      "live_text": True,
    }

    websocket.send_json({"type": "input_text.append", "sequence": 0, "text": "The first fragment "})
    first_json: list[dict[str, object]] = []
    first_pcm = None
    while first_pcm is None:
      kind, frame = _receive_frame(websocket)
      if kind == "bytes":
        first_pcm = frame
      else:
        first_json.append(frame)  # type: ignore[arg-type]
    assert len(cast(bytes, first_pcm)) == 2400
    assert {frame["type"] for frame in first_json} == {"input_text.ack", "started"}
    started = next(frame for frame in first_json if frame["type"] == "started")
    assert started["native_streaming"] is True
    assert started["live_text"] is True

    websocket.send_json(
      {"type": "input_text.append", "sequence": 1, "text": "arrives while speaking."}
    )
    assert websocket.receive_json()["event"] == "input_text.append"
    assert len(websocket.receive_bytes()) == 2400
    websocket.send_json({"type": "input_text.end", "sequence": 2})
    assert websocket.receive_json()["event"] == "input_text.end"
    assert len(websocket.receive_bytes()) == 2400
    finished = websocket.receive_json()
    assert finished["type"] == "finished"
    assert finished["request_id"] == "live-one"
    assert finished["live_text"] is True
    assert finished["generation"]["input_text"] == ("The first fragment arrives while speaking.")

    websocket.send_json({"type": "ping"})
    assert websocket.receive_json() == {"type": "pong"}


def test_live_text_rejects_out_of_order_sequence_without_closing_socket(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Return a stable protocol error and keep the persistent connection usable."""
  with client.websocket_connect("/v1/speech/stream") as websocket:
    websocket.send_json(
      {
        "type": "request.start",
        "request_id": "bad-sequence",
        "profile_id": profile_with_sample["id"],
        "model": "qwen-tts-1.7B",
      }
    )
    assert websocket.receive_json()["type"] == "request.configured"
    websocket.send_json({"type": "input_text.append", "sequence": 4, "text": "Out of order."})
    error = websocket.receive_json()
    assert error["type"] == "error"
    assert error["code"] == "invalid_sequence"
    assert error["request_id"] == "bad-sequence"
    websocket.send_json({"type": "ping"})
    frames = [websocket.receive_json(), websocket.receive_json()]
    assert {frame["type"] for frame in frames} == {"cancelled", "pong"}
