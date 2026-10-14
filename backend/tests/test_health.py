"""Health endpoint tests."""

import pytest


@pytest.mark.asyncio
async def test_health_endpoint_returns_ok(test_client):
    """GET /api/v1/health returns 200 with the expected shape."""
    response = await test_client.get("/api/v1/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "ask-latrobe-policy-db-chatbot"


@pytest.mark.asyncio
async def test_readiness_endpoint_returns_ready(test_client):
    """GET /api/v1/health/ready returns 200."""
    response = await test_client.get("/api/v1/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_health_sets_request_id_header(test_client):
    """Every response carries an X-Request-ID header."""
    response = await test_client.get("/api/v1/health")

    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


@pytest.mark.asyncio
async def test_health_preserves_client_request_id(test_client):
    """If a client sends X-Request-ID, the response echoes it."""
    custom_id = "test-request-id-12345"
    response = await test_client.get(
        "/api/v1/health",
        headers={"X-Request-ID": custom_id},
    )

    assert response.headers.get("x-request-id") == custom_id
