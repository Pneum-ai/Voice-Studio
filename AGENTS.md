# Voice Studio Repository Guidelines

## Mission

Maintain a private, self-hosted voice studio with a public API and bundled web UI.

## Repository Layout

Keep modules focused and names predictable. Prefer this top-level structure:

```text
voice-studio/
  app/
    api/
      routes/
    backends/
      mlx/
      pytorch/
    core/
    domain/
    persistence/
    services/
    ui_dist/  # Generated production UI; tracked and packaged.
    utils/
  docs/
  scripts/
  tests/
    contract/
    integration/
    unit/
  ui/
```

Route modules translate HTTP, WebSocket, and event-stream traffic only.
Business logic belongs in services, database access belongs in persistence,
and inference details belong behind backend interfaces. Put reusable helpers
in narrowly named utility files; do not create catch-all modules.

Use descriptive `snake_case` names for Python modules and functions,
`PascalCase` for classes, and `UPPER_SNAKE_CASE` for constants. Keep directory
and file listings alphabetically organized when ordering has no semantic
meaning.

## Code Standards

- Use Python 3.12, matching the supported range in `pyproject.toml`.
- Indent with 2 spaces. Never use hard tab characters.
- Keep every code line at or below 100 characters.
- Keep every hand-written code file at or below 500 lines, including tests.
- Split a module before it approaches the limits; do not compress readable
  code merely to satisfy them.
- Use double quotes and Ruff-compatible import ordering.
- Add type annotations to all function and method signatures.
- Avoid untyped dictionaries at architectural boundaries. Prefer Pydantic
  models, dataclasses, protocols, enums, and typed aliases.
- Keep dependencies explicit and injected where practical. Avoid new mutable
  module-level singletons.
- Keep blocking model, file, and database work off the async event loop.
- Use named standard-library loggers instead of debug `print` statements.
  Central logcat configuration owns formatting, colors, and upstream records.
- Never use broad exception handling without adding context or preserving the
  intended HTTP or task state.

Generated files such as dependency lockfiles are exempt from the 500-line and
100-character limits and must not be edited manually.

## Simplicity and Test Value

- Treat every change as an opportunity to reduce the total amount of code when
  behavior, clarity, and compatibility can be preserved. Prefer deleting,
  consolidating, and simplifying existing code over adding layers or parallel
  implementations.
- Keep cyclomatic complexity low. Use small single-purpose functions, early
  returns, and straightforward control flow; refactor code whose branches are
  difficult to enumerate or explain.
- Tautological tests are harmful. Do not write tests that merely restate the
  implementation, assert mocked behavior, or reproduce production logic in the
  test. Assert externally observable behavior, meaningful invariants, boundary
  cases, and failure modes so a real defect would cause the test to fail.

## Documentation and Comments

Every Python module, public or private class, function, method, and protocol
must have a useful docstring. Docstrings must explain purpose, important
arguments, return values, raised exceptions, side effects, and compatibility
constraints where relevant.

Use comments to explain non-obvious decisions, platform workarounds, concurrency
rules, protocol quirks, and compatibility requirements. Do not narrate obvious
syntax. Keep documentation accurate when behavior changes.

Public API routes must include OpenAPI-friendly summaries, descriptions,
response models, and documented error responses. Keep the checked-in API
specification authoritative and fail CI when generated OpenAPI drifts from it.

## Bundled UI

The normal launch command serves the bundled UI and API from the same process
and origin:

```bash
uv run voice-open
```

The default dependency group installs MLX on Apple Silicon and Nari's CUDA
stack on Linux/WSL2. No extra is needed for these supported platforms. Keep
CPU compatibility instructions in the README. Model downloads and prewarming
are opt-in; enabling the UI must not change that behavior.

Use `uv run voice-open-api` for API-only startup. Direct `server.py` execution
is also API-only unless passed `--ui`. Keep `/v1`, `/health`, `/docs`, and
`/openapi.json` outside the SPA fallback; unknown API routes must remain errors.
UI mode must preserve API semantics, model behavior, and data paths.

Keep React/Vite source in `ui/`. From the repository root, use
`npm --prefix ui ci` to install frontend tools and `npm --prefix ui run dev`
for the development server, which proxies API and WebSocket traffic to port
13337. Run the backend separately with `uv run voice-open-api`.

`npm --prefix ui run build` replaces `app/ui_dist/`. Commit the generated
assets, including removed old fingerprints, whenever the UI changes. Never
edit the bundle by hand. Release wheels and source archives must contain the
production UI; wheels must exclude frontend source, tests, and `node_modules`.
Neither launch mode requires Node.js. Missing UI assets must produce a short
error explaining the build command.

For UI changes, run frontend lint, relevant tests, and the production build;
check packaging with `python -m scripts.verify_distribution` in the project
environment. Keep documentation screenshots under `docs/screenshots/` and
reference them from the README or other maintained documentation.

The UI should be responsive, accessible, keyboard-friendly, and visually
restrained. Optimize for fast startup and smooth interaction:

- Keep the initial JavaScript bundle small and lazy-load secondary screens.
- Avoid base64 audio and image payloads; use URLs, blobs, and streaming.
- Use event streams for job progress instead of aggressive polling.
- Cancel obsolete requests and release object URLs, audio nodes, and waveform
  resources.
- Decode visible audio lazily for waveform previews; keep scrubbers full-width
  and touch-friendly on mobile.
- Cache fingerprinted static assets and never cache the HTML application shell
  as immutable.
- Keep large model libraries and backend-only dependencies out of the browser
  bundle.
- Test loading, empty, ready, downloading, generating, error, and offline
  states.

Speech model capabilities control the Create screen: Base uses cloned profiles and
reference samples without instructions; CustomVoice 1.7B uses its built-in speakers
and optional instructions without a profile. Never send incompatible voice sources
when switching models. Keep these rules identical across REST and WebSocket routes,
MLX, and NVIDIA/Nari. Preset generations retain their speaker name without a profile
foreign key.

The UI includes voices and reference samples, live speech generation, saved
speech in Library, transcription, model management, jobs, settings, and the
lightweight scene editor. The editor supports clip arrangement, trimming, gain,
fades, undo/redo, local drafts, and WAV export. Avatars and advanced audio
editing remain out of scope.

Reference samples and transcription accept uploads or microphone recordings.
Show reference waveforms, an accessible scrubber, and trim controls before
saving. Use the selected loaded transcription model for automatic reference
transcripts. Release microphone access on cancellation or navigation, and
unlock live audio during the initiating user gesture.

## Dependency Cooldown

Keep the one-week registry cooldown enabled in both package ecosystems:

- `[tool.uv] exclude-newer = "1 week"` for Python.
- `min-release-age=7` in `ui/.npmrc` with npm 11.10 or newer.

Do not bypass these controls silently. A short-lived exception for an urgent
security fix must be explicit and reviewed.

## Storage Performance

The runtime database uses a new versioned format. It does not need to preserve
legacy table or column layouts.

Use SQLite with:

- An explicit schema version and ordered migrations.
- WAL mode, foreign keys, a busy timeout, and deliberate sync settings.
- Relative data paths instead of machine-specific absolute paths.
- Foreign-key cascades that clean up samples with their profile and nullable
  generation references that preserve saved speech after profile deletion.
- Indexes for every list, filter, foreign-key, and recency query used by the
  API or UI.
- Aggregate queries for sample counts; never issue one count query per profile.
- Short transactions and async-safe access that does not block the event loop.

Keep audio files outside SQLite. Store profile names, sample metadata, and
saved generation metadata in the database. A saved generation includes the
input text, profile reference, profile-name snapshot, selected model, language,
seed, instructions, audio metadata, relative audio path, hash, and timestamp.

Do not support, detect, open, or migrate legacy databases. Start with
a new database and keep its schema independent from the reference
implementation.

## Open-Source Readiness

The repository must be understandable and runnable without access to this
monorepo. Include and maintain:

- `README.md` with platform-specific installation and quick-start steps.
- The Apache-2.0 `LICENSE` and required upstream attribution.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and `CHANGELOG.md`.
- `.env.example` containing safe, portable defaults and no secrets.
- Architecture, API compatibility, storage, and backend extension docs.
- Dependency groups or extras that make platform-specific installs explicit.

Do not hard-code personal paths, monorepo-only certificate locations, private
hosts, tokens, or credentials. Local TLS must be opt-in and configurable.

## Security and Data Safety

Voice samples, generated audio, model caches, databases, and environment files
are user data and must remain ignored by Git. Constrain file operations to
configured data directories, limit upload sizes, and avoid unsafe
deserialization of untrusted cache files.

Do not weaken TLS verification globally. Scope any local-development TLS
exception to the client or request that needs it.

## Change Discipline

Keep commits small and imperative, for example:
`api: preserve websocket error envelopes`.
