"""Conversation CRUD endpoint tests (happy paths; isolation is in test_rag_guardrails.py)."""

import pytest


@pytest.fixture
def auth_header(student_token):
    return {"Authorization": f"Bearer {student_token}"}


# ────────────────────────────────────────────────────────────────────────
# Create
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_conversation_with_title(test_client, auth_header):
    response = await test_client.post(
        "/api/v1/conversations",
        json={"title": "My scholarship question"},
        headers=auth_header,
    )

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "My scholarship question"
    assert body["message_count"] == 0
    assert "id" in body


@pytest.mark.asyncio
async def test_create_conversation_without_title_defaults(test_client, auth_header):
    response = await test_client.post(
        "/api/v1/conversations",
        json={},
        headers=auth_header,
    )

    assert response.status_code == 201
    assert response.json()["title"] == "New enquiry"


@pytest.mark.asyncio
async def test_create_conversation_requires_authentication(test_client):
    response = await test_client.post("/api/v1/conversations", json={"title": "x"})

    assert response.status_code == 401


# ────────────────────────────────────────────────────────────────────────
# List
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_conversations_starts_empty(test_client, auth_header):
    response = await test_client.get("/api/v1/conversations", headers=auth_header)

    assert response.status_code == 200
    assert response.json()["conversations"] == []


@pytest.mark.asyncio
async def test_list_conversations_returns_created_conversation(test_client, auth_header):
    await test_client.post(
        "/api/v1/conversations", json={"title": "First"}, headers=auth_header
    )

    response = await test_client.get("/api/v1/conversations", headers=auth_header)

    titles = [c["title"] for c in response.json()["conversations"]]
    assert "First" in titles


@pytest.mark.asyncio
async def test_list_conversations_most_recently_updated_first(test_client, auth_header):
    first = await test_client.post(
        "/api/v1/conversations", json={"title": "Older"}, headers=auth_header
    )
    second = await test_client.post(
        "/api/v1/conversations", json={"title": "Newer"}, headers=auth_header
    )

    # Touch "Older" so it becomes the most recently updated.
    await test_client.patch(
        f"/api/v1/conversations/{first.json()['id']}",
        json={"title": "Older renamed"},
        headers=auth_header,
    )

    response = await test_client.get("/api/v1/conversations", headers=auth_header)
    ids_in_order = [c["id"] for c in response.json()["conversations"]]

    assert ids_in_order[0] == first.json()["id"]
    assert second.json()["id"] in ids_in_order


# ────────────────────────────────────────────────────────────────────────
# Get detail
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_conversation_detail(test_client, auth_header):
    created = await test_client.post(
        "/api/v1/conversations", json={"title": "Detail test"}, headers=auth_header
    )
    conv_id = created.json()["id"]

    response = await test_client.get(
        f"/api/v1/conversations/{conv_id}", headers=auth_header
    )

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == conv_id
    assert body["title"] == "Detail test"
    assert body["messages"] == []


@pytest.mark.asyncio
async def test_get_nonexistent_conversation_returns_404(test_client, auth_header):
    response = await test_client.get("/api/v1/conversations/999999", headers=auth_header)

    assert response.status_code == 404


# ────────────────────────────────────────────────────────────────────────
# Rename
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rename_conversation(test_client, auth_header):
    created = await test_client.post(
        "/api/v1/conversations", json={"title": "Old title"}, headers=auth_header
    )
    conv_id = created.json()["id"]

    response = await test_client.patch(
        f"/api/v1/conversations/{conv_id}",
        json={"title": "New title"},
        headers=auth_header,
    )

    assert response.status_code == 200
    assert response.json()["title"] == "New title"


@pytest.mark.asyncio
async def test_rename_nonexistent_conversation_returns_404(test_client, auth_header):
    response = await test_client.patch(
        "/api/v1/conversations/999999",
        json={"title": "Doesn't matter"},
        headers=auth_header,
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_rename_with_empty_title_returns_422(test_client, auth_header):
    created = await test_client.post(
        "/api/v1/conversations", json={"title": "Keep me"}, headers=auth_header
    )
    conv_id = created.json()["id"]

    response = await test_client.patch(
        f"/api/v1/conversations/{conv_id}",
        json={"title": ""},
        headers=auth_header,
    )

    assert response.status_code == 422


# ────────────────────────────────────────────────────────────────────────
# Delete
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_conversation(test_client, auth_header):
    created = await test_client.post(
        "/api/v1/conversations", json={"title": "To delete"}, headers=auth_header
    )
    conv_id = created.json()["id"]

    response = await test_client.delete(
        f"/api/v1/conversations/{conv_id}", headers=auth_header
    )
    assert response.status_code == 204

    follow_up = await test_client.get(
        f"/api/v1/conversations/{conv_id}", headers=auth_header
    )
    assert follow_up.status_code == 404


@pytest.mark.asyncio
async def test_delete_nonexistent_conversation_returns_404(test_client, auth_header):
    response = await test_client.delete(
        "/api/v1/conversations/999999", headers=auth_header
    )

    assert response.status_code == 404
