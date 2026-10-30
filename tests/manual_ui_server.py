"""Run the production UI against deterministic fake inference backends."""

from __future__ import annotations

from pathlib import Path

import uvicorn

from app.core.config import Settings
from app.main import create_app
from tests.helpers import fake_backends

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
  """Serve a disposable local application for manual browser verification."""
  settings = Settings(
    data_dir=ROOT / ".tmp-ui-qa",
    model_cache_dir=ROOT / ".tmp-ui-models",
    prewarm=False,
    ui_enabled=True,
    ui_dist_dir=ROOT / "ui" / "dist",
  )
  uvicorn.run(
    create_app(settings, fake_backends()),
    host="127.0.0.1",
    port=13338,
    log_level="warning",
  )


if __name__ == "__main__":
  main()
