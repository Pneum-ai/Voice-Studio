"""Profile, sample, and saved-generation API contracts."""

from __future__ import annotations

import io

import librosa
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from tests.helpers import reference_wav_bytes


def test_clean_high_peak_reference_survives_resampling(client: TestClient) -> None:
  """Accept an unclipped 48 kHz WAV even when 24 kHz conversion overshoots full scale."""
  samples = (0.99 * np.sin(2 * np.pi * 6000 * np.arange(96000) / 48000)).astype(np.float32)
  source = io.BytesIO()
  sf.write(source, samples, 48000, format="WAV", subtype="PCM_16")
  decoded, _ = sf.read(io.BytesIO(source.getvalue()), dtype="float32")
  converted = librosa.resample(decoded, orig_sr=48000, target_sr=24000)
  assert np.max(np.abs(decoded)) < 1
  assert np.max(np.abs(converted)) > 1  # The previous validation rejected this clean input.
  profile = client.post("/v1/profiles", json={"name": "High peak voice"}).json()
  added = client.post(
    f"/v1/profiles/{profile['id']}/samples",
    files={"audio": ("clean.wav", source.getvalue())},
    data={"transcript": "Clean source."},
  )
  assert added.status_code == 201, added.text
  result = client.get(added.json()["audio_url"])
  stored, rate = sf.read(io.BytesIO(result.content), dtype="float32")
  assert rate == 24000 and len(stored) == 48000
  assert np.max(np.abs(stored)) <= 0.851


def test_saturated_pcm_reference_reports_actionable_error(client: TestClient) -> None:
  """Detect real flattened PCM peaks even though the file never exceeds full scale."""
  samples = np.clip(1.5 * np.sin(2 * np.pi * 220 * np.arange(96000) / 48000), -1, 1)
  source = io.BytesIO()
  sf.write(source, samples, 48000, format="WAV", subtype="PCM_16")
  profile = client.post("/v1/profiles", json={"name": "Clipped voice"}).json()
  result = client.post(
    f"/v1/profiles/{profile['id']}/samples",
    files={"audio": ("clipped.wav", source.getvalue())},
    data={"transcript": "Clipped source."},
  )
  assert result.status_code == 400
  assert result.json()["code"] == "audio_clipping"
  assert "Lower your microphone input level" in result.json()["detail"]


def test_trimming_discards_clipping_outside_the_reference(client: TestClient) -> None:
  """Users can remove a loud section without the discarded samples blocking the reference."""
  samples = 0.2 * np.sin(2 * np.pi * 220 * np.arange(144000) / 48000)
  samples[:48000] = np.clip(samples[:48000] * 10, -1, 1)
  source = io.BytesIO()
  sf.write(source, samples, 48000, format="WAV", subtype="PCM_16")
  profile = client.post("/v1/profiles", json={"name": "Clean selection"}).json()
  result = client.post(
    f"/v1/profiles/{profile['id']}/samples",
    files={"audio": ("trimmed.wav", source.getvalue())},
    data={"transcript": "Clean words.", "trim_start_seconds": "1", "trim_end_seconds": "3"},
  )
  assert result.status_code == 201, result.text
  assert result.json()["duration_seconds"] == 2


def test_profile_sample_audio_round_trip(client: TestClient) -> None:
  """Create a profile and retrieve its normalized reference sample."""
  created = client.post("/v1/profiles", json={"name": "Sample voice"})
  assert created.status_code == 201
  profile = created.json()
  added = client.post(
    f"/v1/profiles/{profile['id']}/samples",
    files={"audio": ("sample.wav", reference_wav_bytes())},
    data={
      "transcript": "Reference sample transcript.",
      "style_controls": ["emotion_happy", "pace_slow"],
    },
  )
  assert added.status_code == 201, added.text
  sample = added.json()
  assert sample["style_controls"] == ["emotion_happy", "pace_slow"]
  assert sample["audio_url"].endswith(f"/{sample['id']}/audio")
  updated = client.patch(
    f"/v1/profiles/{profile['id']}/samples/{sample['id']}",
    json={
      "transcript": "Reference sample transcript.",
      "style_controls": ["emotion_calm", "energy_low"],
    },
  )
  assert updated.status_code == 200, updated.text
  assert updated.json()["style_controls"] == ["emotion_calm", "energy_low"]
  transcript_only = client.patch(
    f"/v1/profiles/{profile['id']}/samples/{sample['id']}",
    json={"transcript": "Updated exact transcript."},
  )
  assert transcript_only.status_code == 200, transcript_only.text
  assert transcript_only.json()["style_controls"] == ["emotion_calm", "energy_low"]
  audio = client.get(sample["audio_url"])
  assert audio.status_code == 200
  assert audio.headers["content-type"].startswith("audio/wav")


def test_saved_generation_survives_profile_deletion(
  client: TestClient,
  profile_with_sample: dict[str, object],
) -> None:
  """Preserve stored speech and profile-name snapshots after profile deletion."""
  profile_id = str(profile_with_sample["id"])
  created = client.post(
    "/v1/generations",
    json={
      "profile_id": profile_id,
      "text": "Keep this exact generated input.",
      "language": "en",
      "model": "qwen-tts-1.7B",
      "seed": 7,
    },
  )
  assert created.status_code == 201, created.text
  generation = created.json()
  assert generation["input_text"] == "Keep this exact generated input."
  assert generation["profile_name"] == "Test voice"
  searched = client.get(
    "/v1/generations",
    params={"query": "exact generated", "model": "qwen-tts-1.7B"},
  )
  assert [item["id"] for item in searched.json()["items"]] == [generation["id"]]
  assert client.delete(f"/v1/profiles/{profile_id}").status_code == 200

  retained = client.get(f"/v1/generations/{generation['id']}")
  assert retained.status_code == 200
  assert retained.json()["profile_id"] is None
  assert retained.json()["profile_name"] == "Test voice"
  assert client.get(generation["audio_url"]).headers["content-type"].startswith("audio/wav")
