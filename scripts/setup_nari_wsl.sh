#!/usr/bin/env bash
# Install the Nari CUDA runtime into a persistent Linux-owned WSL environment.

set -eu

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

if ! grep -qi microsoft /proc/sys/kernel/osrelease; then
  echo "This setup helper is intended for WSL2." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Install uv inside WSL before running this helper." >&2
  exit 1
fi

VOICE_OPEN_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_OPEN_ENVIRONMENT="${XDG_DATA_HOME:-$HOME/.local/share}/voice-open/venv"
VOICE_OPEN_RUNTIME_STAMP="$VOICE_OPEN_ENVIRONMENT/.voice-open-runtime"

cd "$VOICE_OPEN_ROOT"
export UV_PROJECT_ENVIRONMENT="$VOICE_OPEN_ENVIRONMENT"
uv sync
{
  sha256sum pyproject.toml uv.lock scripts/setup_nari_wsl.sh
} | sha256sum | cut -d " " -f 1 > "$VOICE_OPEN_RUNTIME_STAMP"

echo "Nari WSL environment is ready at $VOICE_OPEN_ENVIRONMENT"
echo "Start it with: bash scripts/run_nari_wsl.sh"
