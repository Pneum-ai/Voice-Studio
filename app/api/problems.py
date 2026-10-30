"""Convert framework and domain failures into one RFC-style problem shape."""

from __future__ import annotations

import logging
import math
import uuid
from typing import Any

from fastapi import HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas.common import ProblemResponse
from app.core.errors import ServiceError

LOGGER = logging.getLogger(__name__)
PROBLEM_MEDIA_TYPE = "application/problem+json"


def request_id(request: Request) -> str:
  """Return the request correlation ID created by middleware."""
  return str(getattr(request.state, "request_id", "unknown"))


def problem_response(
  request: Request,
  *,
  status: int,
  code: str,
  title: str,
  detail: str,
  errors: list[dict[str, Any]] | None = None,
  headers: dict[str, str] | None = None,
) -> JSONResponse:
  """Create a standards-shaped JSON error response."""
  problem = ProblemResponse(
    title=title,
    status=status,
    code=code,
    detail=detail,
    request_id=request_id(request),
    errors=errors,
  )
  return JSONResponse(
    status_code=status,
    content=jsonable_encoder(problem.model_dump(exclude_none=True)),
    media_type=PROBLEM_MEDIA_TYPE,
    headers=headers,
  )


async def handle_service_error(request: Request, error: ServiceError) -> JSONResponse:
  """Render an expected domain error without leaking internal state."""
  headers = None
  if error.retry_after_ms is not None:
    headers = {"Retry-After": str(max(1, math.ceil(error.retry_after_ms / 1000)))}
  return problem_response(
    request,
    status=error.status,
    code=error.code,
    title=error.title,
    detail=error.detail,
    errors=error.errors,
    headers=headers,
  )


async def handle_validation_error(
  request: Request,
  error: RequestValidationError,
) -> JSONResponse:
  """Render Pydantic request errors in the common problem envelope."""
  return problem_response(
    request,
    status=422,
    code="validation_error",
    title="Request validation failed",
    detail="One or more request fields are invalid.",
    errors=jsonable_encoder(error.errors()),
  )


async def handle_http_error(request: Request, error: HTTPException) -> JSONResponse:
  """Render routing and explicit HTTP errors consistently."""
  detail = str(error.detail)
  title = "Resource not found" if error.status_code == 404 else "HTTP error"
  return problem_response(
    request,
    status=error.status_code,
    code="not_found" if error.status_code == 404 else "http_error",
    title=title,
    detail=detail,
  )


async def handle_unexpected_error(request: Request, error: Exception) -> JSONResponse:
  """Log an unexpected failure with context and return a safe response."""
  LOGGER.exception("Unhandled request failure %s", request_id(request), exc_info=error)
  return problem_response(
    request,
    status=500,
    code="internal_error",
    title="Internal server error",
    detail="The request could not be completed.",
  )


def new_request_id(value: str | None) -> str:
  """Accept a constrained caller ID or generate a UUID correlation ID."""
  if value and len(value) <= 128 and value.isprintable():
    return value
  return str(uuid.uuid4())
