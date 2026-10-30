"""Browser-origin policy for the separate main UI."""

from fastapi.testclient import TestClient


def test_loopback_main_ui_can_preflight_v1_requests(client: TestClient) -> None:
  """Allow local HTTP or HTTPS UI ports without allowing arbitrary websites."""
  response = client.options(
    "/v1/profiles",
    headers={
      "Origin": "https://localhost:5173",
      "Access-Control-Request-Method": "GET",
    },
  )

  assert response.status_code == 200
  assert response.headers["access-control-allow-origin"] == ("https://localhost:5173")


def test_external_origins_are_not_allowed_by_default(client: TestClient) -> None:
  """Keep unrelated websites from controlling a local Voice Studio process."""
  response = client.options(
    "/v1/profiles",
    headers={
      "Origin": "https://example.com",
      "Access-Control-Request-Method": "GET",
    },
  )

  assert response.status_code == 400
  assert "access-control-allow-origin" not in response.headers
