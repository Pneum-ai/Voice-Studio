"""Optional bundled-UI routing and cache behavior."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from tests.helpers import fake_backends


def test_ui_mode_serves_shell_without_masking_api_routes(tmp_path: Path) -> None:
  """Serve browser fallbacks while preserving health and API 404 problems."""
  distribution = tmp_path / "ui-dist"
  distribution.mkdir()
  assets = distribution / "assets"
  assets.mkdir()
  (distribution / "index.html").write_text(
    "<!doctype html><title>Test UI</title>",
    encoding="utf-8",
  )
  (assets / "index-fingerprint.js").write_text(
    "document.title = 'Loaded';",
    encoding="utf-8",
  )
  application = create_app(
    Settings(
      data_dir=tmp_path / "data",
      model_cache_dir=tmp_path / "models",
      prewarm=False,
      ui_enabled=True,
      ui_dist_dir=distribution,
    ),
    fake_backends(),
  )
  with TestClient(application) as client:
    assert "Test UI" in client.get("/").text
    assert "Test UI" in client.get("/profiles/example").text
    asset = client.get("/assets/index-fingerprint.js")
    assert asset.status_code == 200
    assert asset.headers["cache-control"] == ("public, max-age=31536000, immutable")
    assert client.get("/health").status_code == 200
    for path in ("/v1/not-a-route", "/livez"):
      missing = client.get(path)
      assert missing.status_code == 404
      assert missing.headers["content-type"].startswith("application/problem+json")
