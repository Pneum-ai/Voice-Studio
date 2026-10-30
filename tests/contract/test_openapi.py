"""Checked OpenAPI drift contract."""

from __future__ import annotations

import json

from scripts.export_openapi import OUTPUT, generated_schema


def test_checked_openapi_matches_routes() -> None:
  """Require route or schema changes to update the public API document."""
  checked = json.loads(OUTPUT.read_text(encoding="utf-8"))
  assert generated_schema() == checked
