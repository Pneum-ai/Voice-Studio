"""OpenAPI customization for accurate problem response media types."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI


def install_openapi_customization(application: FastAPI) -> None:
  """Render documented ProblemResponse bodies as application/problem+json."""
  default_openapi = application.openapi

  def customized_openapi() -> dict[str, Any]:
    """Return the cached schema after normalizing problem response content."""
    schema = default_openapi()
    paths = schema.get("paths", {})
    for path_item in paths.values():
      if not isinstance(path_item, dict):
        continue
      for operation in path_item.values():
        if not isinstance(operation, dict):
          continue
        responses = operation.get("responses", {})
        for response in responses.values():
          if not isinstance(response, dict):
            continue
          content = response.get("content", {})
          json_content = content.get("application/json")
          if not isinstance(json_content, dict):
            continue
          reference = json_content.get("schema", {}).get("$ref", "")
          if str(reference).endswith("/ProblemResponse"):
            content["application/problem+json"] = content.pop("application/json")
    return schema

  application.openapi = customized_openapi
