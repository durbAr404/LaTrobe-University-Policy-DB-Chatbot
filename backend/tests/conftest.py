"""
Shared pytest fixtures for the Ask La Trobe backend.

Design goals:
  - No external services required (mock HF, Gemini, Qdrant).
  - Fast: in-memory SQLite per test.
  - Deterministic: fixed UUIDs, fixed vectors.
  - Isolated per test; compatible with pytest-xdist process-based parallelism.
"""

import os
from collections.abc import AsyncGenerator
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

# Ensure the app uses an in-memory DB before importing anything that
# reads settings.
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("SECRET_KEY", "test-secret-key-must-be-at-least-32-chars")
os.environ.setdefault("GEMINI_API_KEY", "test-gemini-key")
os.environ.setdefault("HF_TOKEN", "test-hf-token")
os.environ.setdefault("QDRANT_MODE", "local")
os.environ.setdefault("QDRANT_PATH", "./storage/qdrant_test")


# ────────────────────────────────────────────────────────────────────────
# Database
# ────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def db_engine():
    """In-memory SQLite engine, fresh per test."""
    from app.db import models  # noqa: F401  (register tables)
    from app.db.database import Base

    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        echo=False,
        future=True,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine) -> AsyncGenerator[AsyncSession, None]:
    """A DB session bound to the in-memory engine. Rolls back on teardown."""
    maker = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with maker() as session:
        yield session
        await session.rollback()


@pytest_asyncio.fixture
async def seeded_user(db_session: AsyncSession):
    """Create a test student user; return its id, email, password, role."""
    from app.core.security import hash_password
    from app.db.models import User

    user = User(
        email="student@test.edu",
        full_name="Test Student",
        password_hash=hash_password("Password123!"),
        role="student",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.commit()

    return {
        "id": user.id,
        "email": user.email,
        "password": "Password123!",
        "role": user.role,
    }


@pytest_asyncio.fixture
async def seeded_admin(db_session: AsyncSession):
    """Create a test admin user; return its id, email, password, role."""
    from app.core.security import hash_password
    from app.db.models import User

    user = User(
        email="admin@test.edu",
        full_name="Test Admin",
        password_hash=hash_password("AdminPass123!"),
        role="admin",
        is_active=True,
    )
    db_session.add(user)
    await db_session.flush()
    await db_session.commit()

    return {
        "id": user.id,
        "email": user.email,
        "password": "AdminPass123!",
        "role": user.role,
    }


# ────────────────────────────────────────────────────────────────────────
# FastAPI test client
# ────────────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def test_client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """FastAPI app with DB dependency overridden to use our test session."""
    from app.api.dependencies import get_rag_service
    from app.core.rate_limit import limiter
    from app.db.database import get_db
    from app.main import app

    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    # Disable rate limiting for tests (otherwise 10/min cap breaks repeated calls).
    previous_limiter_state = limiter.enabled
    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        yield client

    app.dependency_overrides.clear()
    get_rag_service.cache_clear()
    limiter.enabled = previous_limiter_state


# ────────────────────────────────────────────────────────────────────────
# Auth helpers
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def auth_headers_factory():
    """Return a callable that builds Authorization headers."""

    def _make(token: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {token}"}

    return _make


@pytest_asyncio.fixture
async def student_token(test_client: AsyncClient, seeded_user: dict) -> str:
    """Log in as the seeded student and return the JWT."""
    response = await test_client.post(
        "/api/v1/auth/login",
        json={
            "email": seeded_user["email"],
            "password": seeded_user["password"],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest_asyncio.fixture
async def admin_token(test_client: AsyncClient, seeded_admin: dict) -> str:
    """Log in as the seeded admin and return the JWT."""
    response = await test_client.post(
        "/api/v1/auth/login",
        json={
            "email": seeded_admin["email"],
            "password": seeded_admin["password"],
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


# ────────────────────────────────────────────────────────────────────────
# Embedding service (mock)
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_embedding_service():
    """
    Replace EmbeddingService.embed / embed_many with deterministic fakes.

    Vectors are deterministic based on text hash so tests can assert on
    similarity if needed.
    """
    import hashlib

    from app.services.embedding_service import EmbeddingService

    DIM = 1024

    def _fake_vector(text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        values = []
        for i in range(DIM):
            byte = h[i % len(h)]
            values.append((byte / 127.5) - 1.0)
        return values

    async def fake_embed(self, text: str) -> list[float]:
        return _fake_vector(text)

    async def fake_embed_many(self, texts, batch_size=64, max_retries=3):
        return [_fake_vector(t) for t in texts]

    with patch.object(EmbeddingService, "embed", fake_embed), patch.object(
        EmbeddingService, "embed_many", fake_embed_many
    ):
        yield


# ────────────────────────────────────────────────────────────────────────
# Vector store (mock)
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_vector_store():
    """
    Replace VectorStore with an in-memory fake.

    Also patches __init__ so the real embedded Qdrant client is never
    opened — otherwise parallel tests hit a portalocker file-lock error.
    """
    from app.rag.models import RetrievalResult
    from app.services.vector_store import VectorStore

    storage: dict[str, dict] = {}

    def fake_init(self, *args, **kwargs):
        """Skip the real client construction."""
        self.collection_name = "policy_chunks_test"
        self.dimension = 1024
        self.client = None

    async def fake_ensure_collection(self):
        return None

    async def fake_upsert(self, chunks, vectors):
        for chunk, vec in zip(chunks, vectors, strict=True):
            storage[chunk.chunk_id] = {
                "vector": vec,
                "payload": {
                    "chunk_id": chunk.chunk_id,
                    "policy_id": chunk.policy_external_id,
                    "title": chunk.title,
                    "url": chunk.url,
                    "section": chunk.section,
                    "text": chunk.text,
                    "chunk_index": chunk.chunk_index,
                },
            }

    async def fake_search(self, vector, limit):
        results: list[RetrievalResult] = []
        for i, (_chunk_id, entry) in enumerate(list(storage.items())[:limit]):
            p = entry["payload"]
            results.append(
                RetrievalResult(
                    chunk_id=p["chunk_id"],
                    policy_id=p["policy_id"],
                    title=p["title"],
                    section=p["section"],
                    text=p["text"],
                    url=p["url"],
                    vector_score=0.9 - (i * 0.05),
                    lexical_score=0.0,
                    fused_score=0.0,
                )
            )
        return results

    with patch.object(VectorStore, "__init__", fake_init), patch.object(
        VectorStore, "ensure_collection", fake_ensure_collection
    ), patch.object(VectorStore, "upsert", fake_upsert), patch.object(
        VectorStore, "search", fake_search
    ):
        yield storage


# ────────────────────────────────────────────────────────────────────────
# Lexical store (mock)
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_lexical_store():
    """Replace LexicalStore with an in-memory fake."""
    from app.rag.models import RetrievalResult
    from app.services.lexical_store import LexicalStore

    storage: list[dict] = []

    async def fake_ensure_index(self, session):
        return None

    async def fake_rebuild(self, session):
        return None

    async def fake_search(self, session, query, limit):
        tokens = [t.lower() for t in query.split() if t.strip()]
        out: list[RetrievalResult] = []
        for entry in storage:
            text_lower = entry["text"].lower()
            if any(tok in text_lower for tok in tokens):
                out.append(
                    RetrievalResult(
                        chunk_id=entry["chunk_id"],
                        policy_id=entry["policy_id"],
                        title=entry["title"],
                        section=entry["section"],
                        text=entry["text"],
                        url=entry["url"],
                        vector_score=0.0,
                        lexical_score=0.8,
                        fused_score=0.0,
                    )
                )
        return out[:limit]

    with patch.object(LexicalStore, "ensure_index", fake_ensure_index), patch.object(
        LexicalStore, "rebuild", fake_rebuild
    ), patch.object(LexicalStore, "search", fake_search):
        yield storage


# ────────────────────────────────────────────────────────────────────────
# Generation service (mock)
# ────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_generation_service():
    """
    Replace GenerationService.generate with a configurable fake.

    Usage in a test:
        mock_generation_service.response = "Custom [C1] answer"
    """
    from app.services.generation_service import GenerationService

    class _Mock:
        response = (
            "The policy states that you have 30 days to return electronics [C1]. "
            "Contact the IT Service Desk to start the return [C2]."
        )
        calls: list[str] = []

    mock = _Mock()

    async def fake_generate(self, prompt: str) -> str:
        mock.calls.append(prompt)
        return mock.response

    with patch.object(GenerationService, "generate", fake_generate):
        yield mock
