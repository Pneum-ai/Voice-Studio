"""Generate the checked-in OpenAPI document from application routes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.main import create_app

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "openapi-v1.json"


def generated_schema() -> dict[str, Any]:
  """Return the deterministic API-only OpenAPI schema."""
  application = create_app(
    Settings(
      data_dir=ROOT / ".openapi-data",
      prewarm=False,
      ui_enabled=False,
    )
  )
  return application.openapi()


def main() -> None:
  """Write stable, human-diffable OpenAPI JSON."""
  content = json.dumps(
    generated_schema(),
    ensure_ascii=False,
    indent=2,
    sort_keys=True,
  )
  OUTPUT.write_text(f"{content}\n", encoding="utf-8", newline="\n")
  print(f"Wrote {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
  main()
