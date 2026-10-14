"""Admin endpoint tests."""

from unittest.mock import AsyncMock, patch

import pytest


@pytest.mark.asyncio
async def test_reindex_requires_authentication(test_client):
    response = await test_client.post("/api/v1/admin/reindex")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_reindex_forbidden_for_student(test_client, student_token):
    response = await test_client.post(
        "/api/v1/admin/reindex",
        headers={"Authorization": f"Bearer {student_token}"},
    )

    assert response.status_code == 403


@pytest.mark.asyncio
async def test_reindex_succeeds_for_admin(test_client, admin_token):
    """A successful reindex returns the ingestion counts from IngestionService."""
    from app.services.ingestion_service import IngestionService

    fake_ingest = AsyncMock(return_value={"policies": 3, "chunks": 42})

    with patch.object(IngestionService, "__init__", lambda self: None), patch.object(
        IngestionService, "ingest", fake_ingest
    ):
        response = await test_client.post(
            "/api/v1/admin/reindex",
            headers={"Authorization": f"Bearer {admin_token}"},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["policies"] == 3
    assert body["chunks"] == 42
    fake_ingest.assert_awaited_once()
