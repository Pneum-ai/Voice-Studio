"""Static-file helpers for the optional bundled browser application."""

from __future__ import annotations

from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope


class FingerprintedStaticFiles(StaticFiles):
  """Serve Vite fingerprinted assets with an immutable browser cache policy."""

  async def get_response(self, path: str, scope: Scope) -> Response:
    """Return one static response with long-lived immutable caching."""
    response = await super().get_response(path, scope)
    response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response
