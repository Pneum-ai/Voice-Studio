#!/usr/bin/env bash
# Run Voice Studio with the persistent Nari WSL environment.

set -eu

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

VOICE_OPEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_OPEN_ENVIRONMENT="${XDG_DATA_HOME:-$HOME/.local/share}/voice-open/venv"
VOICE_OPEN_RUNTIME_STAMP="$VOICE_OPEN_ENVIRONMENT/.voice-open-runtime"
VOICE_OPEN_STATE="${XDG_STATE_HOME:-$HOME/.local/state}/voice-open"

cd "$VOICE_OPEN_ROOT"
if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv inside WSL before running this helper." >&2
  exit 1
fi
export UV_PROJECT_ENVIRONMENT="$VOICE_OPEN_ENVIRONMENT"
export VOICE_QWEN_ENGINE="${VOICE_QWEN_ENGINE:-nari}"

VOICE_OPEN_RUNTIME_HASH="$({
  sha256sum pyproject.toml uv.lock scripts/setup_nari_wsl.sh
} | sha256sum | cut -d " " -f 1)"
if [ ! -x "$VOICE_OPEN_ENVIRONMENT/bin/python" ] || \
  [ ! -f "$VOICE_OPEN_RUNTIME_STAMP" ] || \
  [ "$(cat "$VOICE_OPEN_RUNTIME_STAMP")" != "$VOICE_OPEN_RUNTIME_HASH" ]; then
  echo "Preparing Voice Open's optimized WSL environment."
  bash scripts/setup_nari_wsl.sh
fi

if [ -z "${VOICE_DATABASE_FILE:-}" ]; then
  VOICE_DATABASE_FILE="$VOICE_OPEN_STATE/voice.db"
  VOICE_SOURCE_DATABASE="${VOICE_DATA_DIR:-$VOICE_OPEN_ROOT/data}/voice.db"
  if [ ! -f "$VOICE_DATABASE_FILE" ] && [ -f "$VOICE_SOURCE_DATABASE" ]; then
    echo "Migrating SQLite database to Linux storage: $VOICE_DATABASE_FILE"
    "$VOICE_OPEN_ENVIRONMENT/bin/python" scripts/migrate_sqlite_database.py \
      "$VOICE_SOURCE_DATABASE" "$VOICE_DATABASE_FILE"
  fi
  export VOICE_DATABASE_FILE
fi

# setup_nari_wsl.sh owns dependency synchronization. Avoid rebuilding the
# editable project over WSL's slower /mnt/c filesystem on every server start.
exec uv run --no-sync --extra cuda -- python -m app.supervisor --ui "$@"
