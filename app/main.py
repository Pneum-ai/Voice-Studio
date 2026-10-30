"""FastAPI application construction and process lifecycle."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app import __version__
from app.api.openapi import install_openapi_customization
from app.api.problems import (
  handle_http_error,
  handle_service_error,
  handle_unexpected_error,
  handle_validation_error,
  new_request_id,
)
from app.api.routes import (
  cache,
  generations,
  jobs,
  model_runtimes,
  models,
  openai_audio,
  profiles,
  speech,
  speech_stream,
  system,
  transcriptions,
)
from app.api.schemas.common import ProblemResponse
from app.api.static_ui import FingerprintedStaticFiles
from app.backends.factory import BackendBundle
from app.core.config import Settings
from app.core.errors import ServiceError
from app.services.container import build_services


def create_app(
  settings: Settings | None = None,
  backend_bundle: BackendBundle | None = None,
) -> FastAPI:
  """Create one independently configurable API application."""
  resolved_settings = settings or Settings()
  resolved_settings.validate_tls_pair()
  resolved_settings.validate_ui_assets()
  services = build_services(resolved_settings, backend_bundle)

  @asynccontextmanager
  async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize and release process-scoped resources."""
    try:
      await services.startup()
      yield
    finally:
      await services.shutdown()

  application = FastAPI(
    title="Pneum.ai Voice Studio",
    summary="Local voice profiles, Qwen speech generation, and model-flexible transcription.",
    version=__version__,
    lifespan=lifespan,
    servers=[{"url": "http://127.0.0.1:13337", "description": "Default local server"}],
    responses={
      422: {
        "model": ProblemResponse,
        "description": "Request validation failed.",
      }
    },
  )
  application.state.services = services
  _install_middleware(application, resolved_settings)
  _install_exception_handlers(application)
  _install_routes(application, resolved_settings)
  install_openapi_customization(application)
  return application


def _install_middleware(application: FastAPI, settings: Settings) -> None:
  """Install request IDs and the constrained browser CORS policy."""

  @application.middleware("http")
  async def attach_request_id(request: Request, call_next: Any) -> Any:
    """Attach a safe correlation ID to request state and response headers."""
    correlation_id = new_request_id(request.headers.get("X-Request-ID"))
    request.state.request_id = correlation_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = correlation_id
    return response

  if settings.cors_origins or settings.cors_origin_regex:
    application.add_middleware(
      CORSMiddleware,
      allow_origins=list(settings.cors_origins),
      allow_origin_regex=settings.cors_origin_regex,
      allow_credentials=False,
      allow_methods=["*"],
      allow_headers=["*"],
    )


def _install_exception_handlers(application: FastAPI) -> None:
  """Register the common problem renderer for all HTTP failure classes."""
  application.add_exception_handler(ServiceError, cast(Any, handle_service_error))
  application.add_exception_handler(
    RequestValidationError,
    cast(Any, handle_validation_error),
  )
  application.add_exception_handler(HTTPException, cast(Any, handle_http_error))
  application.add_exception_handler(Exception, cast(Any, handle_unexpected_error))


def _install_routes(application: FastAPI, settings: Settings) -> None:
  """Install API routers followed by optional static UI fallback routes."""
  for route_module in (
    cache,
    generations,
    jobs,
    model_runtimes,
    models,
    openai_audio,
    profiles,
    speech,
    speech_stream,
    system,
    transcriptions,
  ):
    application.include_router(route_module.router)
  if not settings.ui_enabled:
    return
  assets = settings.ui_dist_dir / "assets"
  if assets.is_dir():
    application.mount(
      "/assets",
      FingerprintedStaticFiles(directory=assets),
      name="ui-assets",
    )

  @application.get("/{browser_path:path}", include_in_schema=False)
  async def ui_fallback(browser_path: str) -> FileResponse:
    """Serve the SPA shell for browser routes but never mask API misses."""
    # Keep the retired probe from returning a misleading 200 with the SPA shell.
    protected = ("v1", "health", "livez", "docs", "openapi.json", "redoc")
    if browser_path in protected or browser_path.startswith(
      tuple(f"{prefix}/" for prefix in protected)
    ):
      raise HTTPException(status_code=404, detail="Route was not found.")
    candidate = _safe_static_candidate(settings.ui_dist_dir, browser_path)
    if candidate is not None and candidate.is_file():
      return FileResponse(candidate)
    return FileResponse(
      settings.ui_dist_dir / "index.html",
      media_type="text/html",
      headers={"Cache-Control": "no-cache"},
    )


def _safe_static_candidate(root: Path, browser_path: str) -> Path | None:
  """Resolve a UI file only when it remains below the configured build root."""
  candidate = (root / browser_path).resolve()
  resolved_root = root.resolve()
  try:
    candidate.relative_to(resolved_root)
  except ValueError:
    return None
  return candidate
