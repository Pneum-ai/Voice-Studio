# Contributing

Thank you for helping improve the local voice service.

## Before Starting

Open an issue for substantial API, storage, or model-output changes. These
surfaces affect client integrations or user data and require agreement before
implementation.

Never include private voice samples, generated speech, model weights,
credentials, or production databases in an issue or pull request.

Contributions are submitted under Apache-2.0 unless explicitly agreed otherwise.
Preserve upstream license and attribution notices when adapting third-party code.

## Development Setup

```bash
uv sync --group dev
npm --prefix ui ci
```

Use npm 11.10 or newer. Python and JavaScript resolution both exclude registry
artifacts published within the last seven days. Do not bypass that cooldown
without documenting the reviewed package and reason.

The default runtime group selects MLX on Apple Silicon and CUDA on Linux/WSL2.

On a CPU-only development host, select the official engine and avoid the default CUDA group:

```bash
uv sync --no-group runtime --extra cpu --group dev
export VOICE_QWEN_ENGINE=official
```

Run the service with isolated test data:

```bash
uv run voice-open-api --data-dir .local-data
```

## Code Standards

- Use two-space indentation and no hard tabs.
- Keep source files at or below 500 lines.
- Keep source lines at or below 100 characters.
- Add type annotations and useful docstrings to every module, class,
  function, method, and protocol.
- Keep HTTP translation in routes, workflows in services, database access in
  persistence modules, and inference details in backends.
- Keep blocking inference, media, and file work off the event loop.
- Do not use pickle or other executable formats for untrusted data.

Run all default gates:

```bash
uv run --group dev -- python -m scripts.verify
```

The verifier runs Ruff formatting and linting, Pyright, source-policy checks, backend tests,
frontend linting and tests, a production UI build, and wheel-content validation.

Run the frontend development server with `npm --prefix ui run dev`. It proxies
REST, SSE, and WebSocket requests to the backend on port `13337`.

The production build is versioned under `app/ui_dist/` and packaged in release wheels. Run
`npm --prefix ui run build` and commit the changed assets whenever the UI changes.

Hardware tests and actual model downloads must remain opt-in.

Tests should catch a concrete failure or protect a user workflow. Prefer observable
results over copied defaults, catalog snapshots, or mock-call inventories. Extend an
existing regression test when it already exercises the same behavior.

## Maintained Scripts

The `scripts/` directory contains installation, release, and deployment tools:

| Script | Purpose |
| --- | --- |
| `run_nari_windows.ps1` | Launch the application from Windows through WSL2. |
| `run_nari_wsl.sh`, `setup_nari_wsl.sh` | Create and reuse the Linux-owned CUDA environment. |
| `migrate_sqlite_database.py` | Copy the current database safely into WSL storage. |
| `verify.py` | Run the repository checks and release build validation. |
| `check_source.py` | Enforce source size, indentation, and line-width limits. |
| `verify_distribution.py` | Check release archives for required files and local data. |
| `export_openapi.py` | Regenerate the public API specification. |
| `benchmark_tts_streaming.py`, `tts_benchmark_metrics.py` | Measure streaming latency and playback stalls. |

To measure a running server, load a speech model and create a voice profile, then run:

```bash
uv run --no-sync -- python -m scripts.benchmark_tts_streaming \
  --profile-id PROFILE_ID --model qwen-tts-1.7B --concurrency 2 \
  --output benchmark-results/streaming.json
```

This uses the development dependencies installed above. Results include latency percentiles,
request failures, and estimated underruns with immediate playback; browser buffering can
change playback results. Use `--help` for load and timing options. Keep one-off probes,
experiment reports, and raw results out of the release tree.

## API Changes

The checked OpenAPI document must match the application:

```bash
uv run -- python -m scripts.export_openapi
uv run -- pytest tests/contract/test_openapi.py
```

Document WebSocket changes in `docs/api-v1.md`, because OpenAPI does not
describe WebSocket frames.

Do not add compatibility aliases to the core API without an explicit
deprecation plan.

## Storage Changes

Add an ordered migration for every schema change. Never repurpose a prior
migration or silently recognize an older application's database.

## Pull Requests

Keep changes focused and use imperative commit subjects, for example:

```text
api: document native streaming capability
storage: index generation model filters
```

Pull requests should state:

- the behavior being replaced or added;
- intentional compatibility differences;
- tests and hardware used;
- data or migration effects; and
- any model-output changes.

By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
