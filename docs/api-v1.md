# API v1 Specification

## Goals

The API exposes four core capabilities:

- Store named voice profiles and reference samples.
- Generate speech with a selected Qwen model.
- Save generated speech with its complete input metadata.
- Transcribe audio with a selected local speech recognition model.
- Download, load, unload, and remove supported models.

The API is designed for the optional bundled UI, direct use by Pneum.ai 
cloud agents, and other third-party applications.

The API uses neutral resource names beneath the Pneum.ai Voice Studio product.

## Runtime

The server defaults to:

```text
http://127.0.0.1:13337
```

CLI flags include:

```text
--host
--port
--data-dir
--ssl-certfile
--ssl-keyfile
--prewarm
--no-prewarm
--shutdown-timeout
--log-level
--ui
```

The default bind address remains local. TLS is opt-in.

Normal startup leaves all model weights unloaded. `--prewarm` downloads missing
weights and loads both configured default models. `--no-prewarm` remains as an
explicit override for deployments that set `VOICE_PREWARM=true`.

Concurrent speech requests are grouped into one generation pass. The
environment settings that control this are:

```text
VOICE_BATCH_ENABLED     default true
VOICE_BATCH_MAX_SIZE    default 8    (1-32)
VOICE_BATCH_WINDOW_MS   default 50   (0-1000)
```

`VOICE_BATCH_WINDOW_MS` is how long the scheduler waits for more requests
before running a batch, so it is also the worst-case latency added to a
request that arrives alone.

## Conventions

- Use `/v1` for product resources.
- Keep `GET /health`, `/docs`, and `/openapi.json` stable and unversioned.
- Use `profile_id` for cloned voices and `speaker` for CustomVoice built-in speakers.
- Use `model` for canonical model IDs.
- Use `instructions` for optional CustomVoice 1.7B delivery instructions; Base rejects them.
- Return UTC timestamps in ISO 8601 format.
- Return collection objects with an `items` field.
- Never expose local filesystem paths.
- Return errors as `application/problem+json`.
- Return `202 Accepted` with a job for asynchronous work.
- Use one SSE job stream for all long-running progress.
- Return synthesized and stored sample audio as `audio/wav`.

## Error Shape

Use `about:blank` until a stable release domain exists:

```json
{
  "type": "about:blank",
  "title": "Model is not ready",
  "status": 409,
  "code": "model_not_ready",
  "detail": "Load qwen-tts-1.7B before generating speech.",
  "request_id": "request UUID"
}
```

Validation details may be added under `errors`, but the top-level shape remains
the same.

## System

### `GET /health`

Return process readiness without loading a model or mutating state:

```json
{
  "status": "ready",
  "version": "1.0.0",
  "backend": "pytorch",
  "device": "cuda",
  "ui_enabled": true,
  "inference": {
    "ready": true,
    "reason": "ready",
    "active": 1,
    "queued": 0,
    "active_limit": 8,
    "active_ceiling": 8,
    "failure_id": null,
    "failure_detail": null,
    "policy_name": "rtx_4090-safe",
    "capture_profile": "rtx_4090-b8",
    "capture_profile_sha256": "<sha256>",
    "capture_fallbacks": 0
  }
}
```

Use `/health` for health checks. It returns `200` when no inference failure is reported,
or `503` with `status: "unavailable"` and inference failure details when the runtime has
failed. It does not load a model; a `200` response does not mean a speech model is loaded.

The root path behaves as follows:

- API-only mode returns small service metadata as JSON.
- `--ui` mode serves the UI application shell.

## Profiles

Profiles store only a name and reference samples. Language, model, instructions,
and other generation options belong to individual requests.

### `GET /v1/profiles`

List profiles, newest first:

```json
{
  "items": [
    {
      "id": "profile UUID",
      "name": "Example voice",
      "sample_count": 1,
      "created_at": "2026-07-29T12:00:00Z",
      "updated_at": "2026-07-29T12:00:00Z"
    }
  ]
}
```

### `POST /v1/profiles`

Create an empty profile:

```json
{
  "name": "Example voice"
}
```

### `GET /v1/profiles/{profile_id}`

Return one profile.

### `PATCH /v1/profiles/{profile_id}`

Rename the profile.

### `DELETE /v1/profiles/{profile_id}`

Delete the profile, its samples, and its prompt cache.

### `GET /v1/profiles/{profile_id}/samples`

List sample metadata.

### `POST /v1/profiles/{profile_id}/samples`

Upload multipart fields:

- `audio`: the reference audio file.
- `transcript`: the exact spoken text.
- `style_controls`: repeated optional canonical delivery conditions.
- `trim_start_seconds`: optional inclusive selection start.
- `trim_end_seconds`: optional exclusive selection end.

The server decodes and stores only the selected interval. Browsers should send
the original upload plus these coordinates instead of allocating a second
client-side PCM or WAV copy.

### `PATCH /v1/profiles/{profile_id}/samples/{sample_id}`

Update the sample transcript and optional `style_controls`. Omitting
`style_controls` preserves existing metadata; sending an empty list clears it.

### `DELETE /v1/profiles/{profile_id}/samples/{sample_id}`

Delete the sample and invalidate the profile prompt cache.

### `GET /v1/profiles/{profile_id}/samples/{sample_id}/audio`

Return the stored resampled mono sample as `audio/wav`.

## Speech Generation

### `POST /v1/audio/speech`

This is the recommended OpenAI-compatible speech endpoint. It accepts the
standard `model`, `input`, `voice`, `instructions`, `response_format`, `speed`,
and `stream_format` names. For Base, `voice` is a local profile ID string or an
OpenAI custom-voice object such as `{"id": "profile UUID"}`. For CustomVoice,
`voice` is a built-in speaker ID such as `"ryan"`. The optional
`language` and `seed` fields are Voice Studio extensions.

Supported response formats are `mp3` (the default), `flac`, `wav`, and `pcm`.
PCM responses stream as mono, signed little-endian PCM16 at 24 kHz and include
`X-Audio-Sample-Rate` and `X-Audio-Channels` headers. The local engine currently
supports only `speed: 1.0` and `stream_format: "audio"`; unsupported values are
rejected instead of silently ignored.

```json
{
  "model": "qwen-tts-1.7B-customvoice",
  "input": "Hello from the local voice service.",
  "voice": "ryan",
  "instructions": "Speak warmly.",
  "response_format": "pcm"
}
```

### `POST /v1/speech`

Legacy Voice Studio request used by existing clients and the bundled UI.
Synthesize and return audio without saving it:

```json
{
  "profile_id": "profile UUID",
  "text": "Hello from the local voice service.",
  "language": "en",
  "model": "qwen-tts-1.7B",
  "seed": 42,
  "instructions": null
}
```

Return `audio/wav`. When a backend produces the complete waveform at once,
include `Content-Length`; do not pretend that slicing a completed WAV provides
lower time to first audio.

The caller selects any compatible Qwen model returned by `GET /v1/models`.
Its `speech` capabilities expose `voice_mode` (`clone` or `builtin`), `speakers`, and
`instructions` support. Transcription models have `speech: null`.

For `qwen-tts-1.7B-customvoice`, omit `profile_id` and pass `speaker` instead:

```json
{
  "model": "qwen-tts-1.7B-customvoice",
  "speaker": "ryan",
  "text": "Hello from the local voice service.",
  "language": "en",
  "instructions": "Speak warmly."
}
```

Speakers are `vivian`, `serena`, `uncle_fu`, `dylan`, `eric`, `ryan`, `aiden`,
`ono_anna`, and `sohee`. Base requires a profile and rejects nonempty instructions
or generation style controls. CustomVoice rejects profile IDs. These rules also
apply to saved generations and both WebSocket generation message types. Saved
CustomVoice audio has `profile_id: null` and the built-in speaker in `profile_name`.

If the selected model is not ready, return a `model_not_ready` problem. The
bundled UI may load the model, follow its job, and retry generation. The server
must not silently begin a large download during the speech request.

### `WS /v1/speech/stream`

Provide one persistent low-latency transport for other real-time
clients.

The client sends:

```json
{
  "type": "generate",
  "request_id": "client request ID",
  "profile_id": "profile UUID",
  "text": "Hello from the local voice service.",
  "language": "en",
  "model": "qwen-tts-1.7B",
  "seed": 42,
  "instructions": null,
  "store": false
}
```

The server responds with a JSON start event:

```json
{
  "type": "started",
  "request_id": "client request ID",
  "encoding": "pcm_s16le",
  "sample_rate": 24000,
  "channels": 1,
  "native_streaming": true
}
```

Binary frames contain raw PCM for that request. A final JSON event completes
the sequence:

```json
{
  "type": "finished",
  "request_id": "client request ID",
  "generation": null
}
```

If `store` is true, `generation` contains the saved generation metadata. The
server must finish staging and committing saved audio before sending
`finished`.

Additional client messages:

- `cancel` with a `request_id` stops queued or active work when supported.
- `ping` receives a `pong` without affecting generation state.

Incremental LLM output can use an additive Nari-style event sequence on the same endpoint.
First configure the fixed voice state without target text:

```json
{
  "type": "request.start",
  "request_id": "client request ID",
  "profile_id": "profile UUID",
  "language": "en",
  "model": "qwen-tts-0.6B",
  "store": false
}
```

Wait for `request.configured`, then send monotonically increasing fragments:

```json
{"type":"input_text.append","sequence":0,"text":"The answer begins here, "}
{"type":"input_text.append","sequence":1,"text":"while later words are still arriving."}
{"type":"input_text.end","sequence":2}
```

Every input event receives `input_text.ack`. The normal `started`, binary PCM, and `finished`
frames are reused, with `live_text: true` added to those events. PCM may arrive before
`input_text.end`. `response.cancel` cancels the active incremental request; the original
`cancel` message with its `request_id` works as well.

Qwen Base requires complete target text in each ICL prefill, unlike CustomVoice's token
continuation lane. Voice Open therefore turns stable appended phrases into ordinary Nari
Base requests, prefetches them concurrently, and drains their native Codec PCM in text order.
This preserves Base quality while overlapping speech with ongoing LLM output.
Input is buffered until at least 24 characters reach a stable word or sentence boundary, or
until the final `input_text.end` event flushes the remaining text.

The connection remains open for multiple sequential requests. Version 1
serializes requests per connection, but request IDs are still mandatory. Error
events include the normal problem fields plus `type: "error"` and
`request_id`.

Requests arriving on *different* connections run concurrently and are batched
into one generation pass when they agree on model. A client that
wants several chunks generated at once should open one connection per
in-flight request. Batching is what keeps generation ahead of real time:
generating one chunk alone runs slower than the audio it produces, while a
batch of eight runs several times faster than real time.

Because a batch is one sampling pass, audio for a given text may differ
between a batched and an unbatched run exactly as it differs between two
seeds. Requests that pass an explicit `seed` are never batched, so seeded
generation stays reproducible.

A `cancel` that arrives after a request has been dispatched to the model
completes the generation and discards its audio.

Backends expose whether true streaming is available:

- Qwen 0.6B and 1.7B Base expose native streaming when the Nari CUDA runtime is active.
- Nari produces PCM from incremental Codec state; completed waveforms are never sliced to
  simulate streaming.

- Streaming backends emit PCM frames as audio becomes available and bypass
  batching.
- Non-streaming backends emit one PCM frame after full inference.

## Saved Generations

### `GET /v1/generations`

List saved generations newest first. Support cursor pagination and optional
`profile_id`, `model`, and text-query filters.

### `POST /v1/generations`

Use the same input as `POST /v1/speech`, save the WAV and complete input
metadata, and return:

```json
{
  "id": "generation UUID",
  "profile_id": "profile UUID",
  "profile_name": "Example voice",
  "input_text": "Hello from the local voice service.",
  "language": "en",
  "model": "qwen-tts-1.7B",
  "seed": 42,
  "instructions": null,
  "duration_seconds": 2.4,
  "sample_rate": 24000,
  "audio_url": "/v1/generations/generation UUID/audio",
  "created_at": "2026-07-29T12:00:00Z"
}
```

### `GET /v1/generations/{generation_id}`

Return saved input and audio metadata.

### `DELETE /v1/generations/{generation_id}`

Delete the database row and generated WAV.

### `GET /v1/generations/{generation_id}/audio`

Return the saved audio as `audio/wav`.

## Transcription

### `POST /v1/audio/transcriptions`

This is the recommended OpenAI-compatible transcription endpoint. Upload the
standard multipart fields `file`, `model`, optional `language`, optional
`prompt`, optional `temperature`, and optional `response_format`.

`response_format` may be `json` (the default), `text`, `srt`, `vtt`, or
`verbose_json`. Because the local backend currently returns whole-file text
rather than timestamped tokens, SRT and VTT contain one whole-file cue and the
verbose response exposes empty `segments` and `words` arrays.

### `POST /v1/transcriptions`

Legacy Voice Studio upload contract. Upload multipart fields:

- `audio`: the audio file to transcribe.
- `model`: the selected transcription model ID.
- `language`: optional language hint.
- `context`: optional vocabulary or recognition context when supported.
- `initial_prompt`: deprecated compatibility alias for `context`.
- `trim_start_seconds`: optional inclusive selection start.
- `trim_end_seconds`: optional exclusive selection end.

When trim coordinates are present, the selected model receives only that
server-side selection and the response duration describes the selected interval.

Return:

```json
{
  "text": "Transcribed text.",
  "duration_seconds": 5.5,
  "language": "en",
  "requested_language": "en",
  "model": "whisper-base"
}
```

The caller selects any compatible transcription model returned by
`GET /v1/models`. Language and recognition-context inputs are validated against
the selected model's advertised capabilities.
Return `model_not_ready` instead of silently starting a download.
Version 1 accepts source audio up to 60 minutes and the configured upload-byte
limit.

## Models

### `GET /v1/models`

Return every speech and transcription model with:

- `id`
- `family`
- `provider`
- `kind`, either `tts` or `transcription`
- `download_state`
- `load_state`
- `size_bytes`
- `backend_supported`
- `active_job_id`
- `native_streaming`
- `transcription`, containing supported languages, language-selection mode,
  context support, timestamps, streaming, translation, and audio limits

The bundled UI groups models by kind and requires an explicit selection for
generation and transcription.

The transcription catalog includes the existing Whisper IDs, Qwen3-ASR 0.6B
and 1.7B, NVIDIA Canary 1B v2, and NVIDIA Parakeet TDT 0.6B v3. Parakeet uses
automatic language detection and rejects language/context inputs. Canary
requires a supported source language. Qwen3-ASR and Whisper accept optional
recognition context. Optional runtimes that are not installed are represented
with `backend_supported: false`.

### `GET /v1/models/{model_id}`

Return one model.

### `PATCH /v1/models/{model_id}`

Set a desired state:

```json
{
  "state": "loaded"
}
```

Valid states are:

- `loaded`: download if needed, then load.
- `downloaded`: ensure files exist, but do not keep the model loaded.
- `absent`: unload and remove cached model files.

Return the lifecycle job with `202 Accepted`. A no-op transition still uses a
short job so clients can follow one consistent contract.

Only one speech model and one transcription model need to remain loaded at a time.
Loading another model of the same kind may unload the previous one.

### `POST /v1/model-runtimes/install-all`

Install the optional Qwen3-ASR and NVIDIA NeMo dependencies as a retained
background job. Return `202 Accepted` with the standard job object. A second
request while installation is active returns `409 Conflict`.

Normal server startup already provisions these runtimes automatically. This
endpoint remains available to API clients that intentionally request a manual
reinstall after startup.

The managed Qwen3-ASR and Python 3.12 NVIDIA ASR environments live below the
configured data directory. The endpoint installs dependencies only; callers still use
`PATCH /v1/models/{model_id}` with `state: loaded` to download and load weights.
After the installation job completes, clients should refresh `GET /v1/models`
to observe updated `backend_supported` values. An incompatible optional runtime
remains unsupported and is never replaced with a different model.

## Jobs and Progress

### `GET /v1/jobs`

List active jobs. Completed jobs may remain available for a short retention
period.

### `GET /v1/jobs/{job_id}`

Return one download or model-load job.

### `GET /v1/jobs/{job_id}/events`

Stream the same job representation using SSE. Send heartbeat comments while
idle and one final `complete`, `cancelled`, or `failed` event.

Every job uses a generated job ID. Model names are attributes, not job IDs.

## Cache

### `DELETE /v1/cache/profile-prompts`

Clear in-memory prompts, persisted prompts, and combined reference audio.
Return the total number of deleted cache files.

## Intentionally Removed from v1

- Avatars and image processing.
- Profile descriptions and profile-level language.
- `POST /shutdown`.
- Old `/generate`, `/generate/stream`, and `/audio/{generation_id}` paths.
- Old `/ws/tts` envelopes and WAV-over-WebSocket framing.
- Model-specific progress URLs.
- Separate load, unload, download, and delete actions.
- Filesystem paths in JSON responses.

## UI Routes

When `--ui` is enabled:

- Serve the application shell from `/`.
- Serve fingerprinted assets from `/assets/`.
- Apply the SPA fallback only to unknown browser routes.
- Never apply the SPA fallback to `/v1`, `/health`, `/docs`, or
  `/openapi.json`.
- Use same-origin API requests by default.

The bundled UI supports:

- Creating and renaming profiles.
- Uploading, reviewing, editing, and deleting samples.
- Automatically transcribing selected sample files into an editable exact
  transcript.
- Displaying decoded full-width waveforms with keyboard and touch scrubbers.
- Selecting a Qwen model and generating speech.
- Saving, browsing, playing, and deleting generated speech.
- Selecting a transcription model and transcribing audio.
- Downloading models and following job progress.
- Switching light, dark, and system themes.
- Inspecting service health and clearing derived prompt caches.

`whisper-turbo` remains the compatibility default while the expanded model
catalog is benchmarked. Configure another supported default with
`VOICE_DEFAULT_TRANSCRIPTION_MODEL`. Without `--prewarm`, the selected model must
be loaded through the lifecycle API before inference; selecting a sample never
starts a hidden download.

The UI build and API OpenAPI document are versioned together in release
artifacts.
