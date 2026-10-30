# Architecture

## Goals

The service keeps transport, workflows, persistence, and inference
independent so each can evolve without recreating the tightly coupled
reference application.

The process has no mutable application singleton. `create_app` builds one
explicit service graph, which enables isolated tests and multiple configured
instances in one Python process.

## Layers

```text
HTTP / WebSocket routes
          |
      services
       /     \
persistence  backend protocols
     |          /       \
  SQLite     PyTorch     MLX
```

- `app/api/routes` translates HTTP, SSE, and WebSocket traffic.
- `app/api/schemas` defines public request and response shapes.
- `app/services` owns profile, model, prompt, speech, generation, and
  transcription workflows.
- `app/persistence` owns mappings, migrations, and query behavior.
- `app/backends` isolates platform and inference-library details.
- `app/utils` contains narrow audio, upload, path, and cache helpers.

Routes never manipulate SQLAlchemy records or inference models directly.

## Logging

`app/core/logging.py` routes standard application, Uvicorn, warning, and
third-party records through one logcat-style console handler. Modules continue
using named standard-library loggers so dependencies remain interoperable.
Logger tags are shortened centrally, multiline records remain aligned, and
colors are enabled only for compatible interactive terminals.

## Application Lifecycle

Startup performs these operations:

1. Validate TLS and optional UI configuration.
2. Create private runtime directories.
3. Apply ordered SQLite migrations.
4. Run SQLite integrity and foreign-key checks.
5. Download missing weights and load both configured default models unless
   startup prewarming is disabled.

Shutdown cancels lifecycle jobs, unloads both model kinds, and disposes the
async engine in that order. Blocking native inference uses cancellation-friendly
daemon workers so a cancelled operation cannot keep the process alive. Uvicorn
drains requests for a bounded interval before lifespan cleanup begins.

Replacing one Qwen TTS size with the other uses a process-level CUDA reset when the
standard `app` supervisor is active. The target model and lifecycle-job identity are
atomically persisted before exit. The replacement worker restores that job, loads TTS
before STT, marks the job complete, and keeps the selected model for future starts. A
direct, unsupervised server instead unloads STT and TTS, runs cyclic garbage collection,
then restores the same clean load order.

## Speech Model Capabilities

The catalog exposes each speech model's voice source, built-in speakers, and instruction
support. SpeechService validates the source before resolving a cloned reference or creating
a typed built-in speaker prompt. Base rejects delivery instructions; CustomVoice rejects
cloned profiles. Saved preset speech has a null profile ID and a speaker-name snapshot.

Both sources use the same serving queue, playback controls, and audio transports. MLX
prepares per-row reference or preset conditioning before its shared continuous decode loop.
Nari uses its native CustomVoice request layout and the existing Base cloning extension.

## Inference Concurrency

Only one speech model and one transcription model remain loaded at a time. Lifecycle jobs
serialize state transitions per model kind. Speech and transcription services
use separate lifecycle guards. The STT registry owns Whisper, Qwen3-ASR, and
  NVIDIA backends and unloads the active family before loading another one.

The persistent WebSocket runs generation in a task while its receive loop
continues accepting pings and cancellation. Each connection has at most one
active generation. Awaiting every binary send applies socket backpressure.

MLX live and saved speech enter the same backend-owned queue; the service batcher
uses a shared model lease instead of adding a second batching window. One persistent
GPU worker advances a continuous session with at most eight active rows. Each row
retains independent Talker attention, codec convolution, and decoder attention state.
Mixed voices and languages can share a pass. Explicit seeds introduce a FIFO barrier
and run alone through the same session implementation.

The MLX queue accepts at most 32 outstanding requests. Queued work has a 30-second
deadline; total request time is limited to 300 seconds. Each consumer buffers at most
32 audio chunks. A stalled consumer fails explicitly without blocking healthy streams
or saving partial audio as complete. Cancellation and unload release native state.
Playback feedback governs prefill admission, using a four-frame decode cadence,
320 ms startup buffer after leading-silence trimming, and a 100 ms safety margin.
Talker weights are quantized to 8 bits in memory and residual prediction is compiled.
Normalized references are bounded to sixteen per session; speaker embeddings are
cached automatically for 0.6B. Downloaded weights are unchanged.

Speech reserves the GPU worker while active audio is being generated. When transcription
or model work waits, the engine stops admitting newcomers, drains submitted speech,
and releases the reservation. This avoids a transcription call interrupting existing
playback while preventing new speech from indefinitely starving transcription.

On supported Linux/WSL2 CUDA hosts, PyTorch Qwen Base checkpoints run
through the vendored Nari engine. Base-specific continuous speaker and ICL embeddings enter
Nari's captured Talker prefill; the paged Talker decode, batched Code Predictor, unified
request scheduler, and stateful Codec remain on Nari's execution path. Reference Codec frames
seed decoder state but their PCM window is hidden, so every client chunk contains only newly
  synthesized speech. Multiple native streams using the same model bypass the legacy generation
  mutex and are scheduled together by Nari.

The official Qwen runtime remains the portable complete-waveform fallback when Nari is
unavailable.

## Storage

SQLite stores metadata only. Normalized samples and saved generations use
relative file paths below the configured data directory.

File workflows follow this order:

1. Decode and validate outside a database transaction.
2. Write to a temporary file beside its destination.
3. Atomically promote the file.
4. Commit the metadata row.
5. Remove the promoted file if the commit fails.

Deleting a profile cascades to samples and sets saved generation references
to null. A name snapshot preserves generation display context.

Model lifecycle downloads query repository byte metadata and sample complete plus partial
Hugging Face cache blobs every half second. The model service maps those byte counts into
bounded job progress so later load and finalization phases remain distinct.

## Prompt Cache

The prompt cache key includes the selected model, sample IDs, audio hashes,
and transcripts. Multi-sample profiles concatenate normalized 24 kHz audio in
sample order and join transcripts in the same order.

PyTorch prompt tensors use `safetensors` plus validated JSON metadata. The
service never loads prompt pickle files. MLX prompt objects reference the
derived WAV and are reconstructed cheaply.

Any sample-set or transcript mutation invalidates all derived artifacts for
that profile.

## API Contracts

REST resources use `/v1`. Health and API documentation remain unversioned.
All JSON failures use one problem shape and request correlation ID.

OpenAPI is generated from routes and checked into `docs/openapi-v1.json`.
Contract tests fail when it drifts. WebSocket frames are specified separately
in `docs/api-v1.md`.

## Optional UI

API-only startup does not import frontend tooling. `--ui` validates a
production build, serves fingerprinted assets, and applies the SPA fallback
only to browser routes.

The React application keeps REST, SSE, persistent WebSocket, audio conversion,
theme, and application-state concerns in separate modules. Secondary screens
are lazy-loaded. Browser audio uses object URLs, saved audio uses direct
same-origin URLs, job progress follows SSE, and live TTS keeps one reusable
WebSocket connection.

Vite provides the development server and builds fingerprinted assets. FastAPI
serves those assets only when `--ui` is enabled, never masks API misses with
the SPA shell, and sends the HTML shell with `Cache-Control: no-cache`.
