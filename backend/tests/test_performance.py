"""
Performance / scalability tests.

These are not millisecond-precise benchmarks (CI machines vary too much
for that) — they exist to catch algorithmic regressions: an accidental
O(n^2) loop, a lock that serializes concurrent requests, a query that
turns into N+1. Thresholds are deliberately generous (10-50x realistic
cost) so they fail only on a real regression, not CI jitter.

All are marked ``slow`` (see pyproject.toml) so they can be excluded with
``pytest -m "not slow"`` when iterating quickly.
"""

import asyncio
import time

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.rag.chunker import PolicyChunker, RawPolicy
from app.rag.citation import extract_citation_ids
from app.rag.models import RetrievalResult
from app.rag.scoring import reciprocal_rank_fusion
from app.services.reranking_service import RerankingService
from app.services.validation_service import ValidationService

pytestmark = pytest.mark.slow


def make_result(chunk_id: str, text: str = "chunk text") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        policy_id="p1",
        title="T",
        section="S",
        text=text,
        url="https://example.test",
        vector_score=0.5,
        lexical_score=0.5,
        fused_score=0.0,
    )


# ────────────────────────────────────────────────────────────────────────
# Chunker
# ────────────────────────────────────────────────────────────────────────


def test_chunker_handles_large_policy_within_time_budget():
    """A large multi-section policy (~1MB) should chunk in well under a second."""
    sections = [
        {
            "heading": f"Section {i}",
            "text": ("This is a representative policy sentence. " * 150),
        }
        for i in range(200)
    ]
    policy = RawPolicy(
        external_id="perf-policy",
        title="Large Policy",
        url="https://example.test/large",
        crawled_at="2026-01-01T00:00:00Z",
        sections=sections,
    )
    chunker = PolicyChunker()

    started = time.perf_counter()
    chunks = chunker.chunk(policy)
    elapsed = time.perf_counter() - started

    assert len(chunks) > 0
    assert elapsed < 1.0, f"Chunking took {elapsed:.3f}s — expected < 1.0s"


def test_chunker_cost_scales_linearly_not_quadratically():
    """Doubling the input should roughly double the time, not quadruple it."""

    def build_policy(section_count: int) -> RawPolicy:
        sections = [
            {"heading": f"S{i}", "text": "Sentence content here. " * 80}
            for i in range(section_count)
        ]
        return RawPolicy(
            external_id="perf-scale",
            title="Scale Policy",
            url="https://example.test/scale",
            crawled_at=None,
            sections=sections,
        )

    chunker = PolicyChunker()

    small = build_policy(50)
    large = build_policy(400)  # 8x the sections

    # Warm up (first call pays import/JIT-adjacent costs).
    chunker.chunk(build_policy(10))

    start_small = time.perf_counter()
    chunker.chunk(small)
    small_elapsed = time.perf_counter() - start_small

    start_large = time.perf_counter()
    chunker.chunk(large)
    large_elapsed = time.perf_counter() - start_large

    # 8x the input should cost well under 8^2 = 64x the time. Allow a very
    # generous 40x to absorb noise while still catching O(n^2) behavior.
    assert large_elapsed < max(small_elapsed * 40, 0.5), (
        f"small={small_elapsed:.4f}s large={large_elapsed:.4f}s — "
        "cost looks superlinear"
    )


# ────────────────────────────────────────────────────────────────────────
# Reciprocal Rank Fusion
# ────────────────────────────────────────────────────────────────────────


def test_rrf_fusion_scales_to_thousands_of_results():
    vector_results = [make_result(f"v{i}") for i in range(5000)]
    lexical_results = [make_result(f"v{i}") if i % 2 == 0 else make_result(f"l{i}") for i in range(5000)]

    started = time.perf_counter()
    fused = reciprocal_rank_fusion(vector_results, lexical_results, k=60)
    elapsed = time.perf_counter() - started

    assert len(fused) > 0
    scores = [r.fused_score for r in fused]
    assert scores == sorted(scores, reverse=True)
    assert elapsed < 1.0, f"RRF fusion took {elapsed:.3f}s for 10k inputs — expected < 1.0s"


# ────────────────────────────────────────────────────────────────────────
# Reranking (BM25)
# ────────────────────────────────────────────────────────────────────────


def test_reranking_handles_large_candidate_pool_within_time_budget():
    service = RerankingService(bm25_weight=0.4)
    candidates = [
        make_result(
            f"c{i}",
            f"Policy chunk number {i} discusses enrolment, leave, and scholarship "
            "rules for graduate and undergraduate students across all faculties.",
        )
        for i in range(500)
    ]

    started = time.perf_counter()
    reranked = service.rerank("scholarship rules for graduate students", candidates)
    elapsed = time.perf_counter() - started

    assert len(reranked) <= service.settings.final_context_chunks
    assert elapsed < 2.0, f"Reranking 500 candidates took {elapsed:.3f}s — expected < 2.0s"


# ────────────────────────────────────────────────────────────────────────
# Citation extraction / validation
# ────────────────────────────────────────────────────────────────────────


def test_citation_extraction_scales_with_long_answer():
    answer = " ".join(f"Fact number {i} [C{(i % 50) + 1}]." for i in range(2000))

    started = time.perf_counter()
    ids = extract_citation_ids(answer)
    elapsed = time.perf_counter() - started

    assert len(ids) == 50  # deduplicated
    assert elapsed < 0.2, f"Citation extraction took {elapsed:.3f}s — expected < 0.2s"


def test_validation_service_handles_many_citations_quickly():
    service = ValidationService()
    retrieved = [make_result(f"c{i}") for i in range(50)]
    answer = " ".join(f"Fact [C{i + 1}]." for i in range(50))

    started = time.perf_counter()
    service.validate(answer, retrieved)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.2, f"Validation took {elapsed:.3f}s — expected < 0.2s"


# ────────────────────────────────────────────────────────────────────────
# API concurrency (catches accidental serialization / blocking)
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_endpoint_handles_concurrent_requests_quickly(test_client):
    """
    50 concurrent health checks should resolve promptly. A large blowup here
    would indicate something (middleware, logging, a lock) is serializing
    requests that should be independent.
    """
    concurrency = 50

    started = time.perf_counter()
    responses = await asyncio.gather(
        *[test_client.get("/api/v1/health") for _ in range(concurrency)]
    )
    elapsed = time.perf_counter() - started

    assert all(r.status_code == 200 for r in responses)
    assert elapsed < 3.0, f"{concurrency} concurrent health checks took {elapsed:.3f}s"


@pytest_asyncio.fixture
async def concurrent_client(db_engine):
    """
    A test client where every request gets its OWN AsyncSession, same as
    production's ``get_db`` dependency.

    The shared ``test_client``/``db_session`` fixtures intentionally hand
    every request the SAME session object for simplicity — convenient for
    sequential tests, but SQLAlchemy's AsyncSession is explicitly not
    safe to use concurrently from multiple coroutines. Reusing it here
    would produce PK collisions that are an artifact of the shared test
    fixture, not a real concurrency bug in the app.
    """
    from app.api.dependencies import get_rag_service
    from app.core.rate_limit import limiter
    from app.db.database import get_db
    from app.main import app

    session_factory = async_sessionmaker(
        bind=db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db
    previous_limiter_state = limiter.enabled
    limiter.enabled = False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client

    app.dependency_overrides.clear()
    get_rag_service.cache_clear()
    limiter.enabled = previous_limiter_state


@pytest_asyncio.fixture
async def concurrent_student_token(concurrent_client, db_engine):
    """Seed a user via a throwaway session, then log in through the real endpoint."""
    from app.core.security import hash_password
    from app.db.models import User

    session_factory = async_sessionmaker(
        bind=db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with session_factory() as session:
        session.add(
            User(
                email="perf-student@test.edu",
                full_name="Perf Student",
                password_hash=hash_password("Password123!"),
                role="student",
                is_active=True,
            )
        )
        await session.commit()

    response = await concurrent_client.post(
        "/api/v1/auth/login",
        json={"email": "perf-student@test.edu", "password": "Password123!"},
    )
    assert response.status_code == 200, response.text
    return response.json()["access_token"]


@pytest.mark.asyncio
async def test_chat_pipeline_handles_concurrent_requests(
    concurrent_client,
    concurrent_student_token,
    mock_embedding_service,
    mock_vector_store,
    mock_lexical_store,
    mock_generation_service,
):
    """
    Several truly concurrent chat requests (own DB session each, like
    production; external services mocked out) should all complete and
    resolve promptly — a regression guard against accidental
    serialization in the RAG pipeline (e.g. a shared non-async lock or a
    per-request service being rebuilt expensively).
    """
    test_client = concurrent_client
    student_token = concurrent_student_token

    mock_vector_store["c1"] = {
        "vector": [0.1] * 1024,
        "payload": {
            "chunk_id": "c1",
            "policy_id": "policy-1",
            "title": "Return Policy",
            "url": "https://example.test/policy/return",
            "section": "Returns",
            "text": "You can return electronics within 30 days.",
            "chunk_index": 0,
        },
    }
    mock_generation_service.response = "You can return electronics within 30 days [C1]."

    concurrency = 10
    headers = {"Authorization": f"Bearer {student_token}"}

    async def send_one(i: int):
        return await test_client.post(
            "/api/v1/chat",
            json={
                "query": f"How do I return electronics? (request {i})",
                "conversation": [],
                "conversation_id": None,
            },
            headers=headers,
        )

    started = time.perf_counter()
    responses = await asyncio.gather(*[send_one(i) for i in range(concurrency)])
    elapsed = time.perf_counter() - started

    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["grounded"] is True for r in responses)
    assert elapsed < 5.0, f"{concurrency} concurrent chat requests took {elapsed:.3f}s"
