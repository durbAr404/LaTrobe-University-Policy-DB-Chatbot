"""RerankingService (BM25 blending) tests."""

import pytest

from app.rag.models import RetrievalResult
from app.services.reranking_service import RerankingService


def make_result(chunk_id: str, text: str, fused_score: float = 0.5) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        policy_id="p1",
        title="T",
        section="S",
        text=text,
        url="https://example.test",
        vector_score=0.5,
        lexical_score=0.5,
        fused_score=fused_score,
    )


@pytest.fixture
def service():
    return RerankingService(bm25_weight=0.4)


# ────────────────────────────────────────────────────────────────────────
# Construction
# ────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("weight", [-0.1, 1.1, 2.0, -5.0])
def test_rejects_out_of_range_bm25_weight(weight):
    with pytest.raises(ValueError):
        RerankingService(bm25_weight=weight)


@pytest.mark.parametrize("weight", [0.0, 0.5, 1.0])
def test_accepts_boundary_bm25_weights(weight):
    RerankingService(bm25_weight=weight)  # should not raise


# ────────────────────────────────────────────────────────────────────────
# rerank
# ────────────────────────────────────────────────────────────────────────


def test_rerank_empty_candidates_returns_empty(service):
    assert service.rerank("anything", []) == []


def test_rerank_empty_query_returns_candidates_unchanged(service):
    candidates = [make_result("a", "Some text"), make_result("b", "Other text")]

    result = service.rerank("   ", candidates)

    assert result == list(candidates)


def test_rerank_promotes_lexically_matching_chunk(service):
    """
    A chunk that actually contains the query terms should outrank ones that
    don't. Needs >=3 candidates: with exactly 2 docs, BM25's IDF for any
    term appearing in exactly one of them degenerates to 0 (N=2, n=1 is
    the classic zero-IDF case), masking the effect we're testing.
    """
    candidates = [
        make_result("off_topic", "This chunk discusses library opening hours."),
        make_result(
            "on_topic",
            "Scholarship refund policy students may request a refund within 30 days.",
        ),
        make_result("neutral", "Parking permits are issued at the campus security office."),
    ]

    reranked = service.rerank("scholarship refund policy", candidates)

    assert reranked[0].chunk_id == "on_topic"


def test_rerank_preserves_all_candidates(service):
    candidates = [make_result(f"c{i}", f"text number {i}") for i in range(5)]

    reranked = service.rerank("text number", candidates)

    assert {r.chunk_id for r in reranked} == {c.chunk_id for c in candidates}


def test_rerank_respects_top_k(service):
    candidates = [make_result(f"c{i}", f"policy text {i}") for i in range(10)]

    reranked = service.rerank("policy text", candidates, top_k=3)

    assert len(reranked) == 3


def test_rerank_sorts_by_rerank_score_descending(service):
    candidates = [
        make_result("a", "irrelevant filler content about parking permits"),
        make_result("b", "graduate scholarship application deadline policy"),
        make_result("c", "campus library opening hours and locations"),
    ]

    reranked = service.rerank("graduate scholarship deadline", candidates)

    scores = [r.rerank_score for r in reranked]
    assert scores == sorted(scores, reverse=True)
    assert len(set(scores)) > 1


def test_rerank_does_not_mutate_input_candidates(service):
    candidates = [make_result("a", "scholarship policy text")]
    original_score = candidates[0].rerank_score

    service.rerank("scholarship policy", candidates)

    assert candidates[0].rerank_score == original_score


def test_rerank_bm25_weight_zero_keeps_fused_order(service):
    """With bm25_weight=0, rerank_score should equal the original fused_score."""
    zero_weight_service = RerankingService(bm25_weight=0.0)
    candidates = [
        make_result("a", "totally unrelated text", fused_score=0.9),
        make_result("b", "also unrelated text", fused_score=0.1),
    ]

    reranked = zero_weight_service.rerank("query terms not in corpus", candidates)

    by_id = {r.chunk_id: r.rerank_score for r in reranked}
    assert by_id["a"] == pytest.approx(0.9)
    assert by_id["b"] == pytest.approx(0.1)


# ────────────────────────────────────────────────────────────────────────
# _tokenize
# ────────────────────────────────────────────────────────────────────────


def test_tokenize_lowercases_and_splits():
    assert RerankingService._tokenize("Hello World") == ["hello", "world"]


def test_tokenize_empty_string_returns_empty_list():
    assert RerankingService._tokenize("") == []


def test_tokenize_collapses_whitespace():
    assert RerankingService._tokenize("a   b\tc") == ["a", "b", "c"]


# ────────────────────────────────────────────────────────────────────────
# _min_max_normalize
# ────────────────────────────────────────────────────────────────────────


def test_min_max_normalize_scales_to_unit_range():
    result = RerankingService._min_max_normalize([0.0, 5.0, 10.0])

    assert result == [0.0, 0.5, 1.0]


def test_min_max_normalize_handles_all_equal_values():
    result = RerankingService._min_max_normalize([3.0, 3.0, 3.0])

    assert result == [1.0, 1.0, 1.0]


def test_min_max_normalize_handles_empty_input():
    assert RerankingService._min_max_normalize([]) == []


def test_min_max_normalize_handles_negative_values():
    result = RerankingService._min_max_normalize([-10.0, 0.0, 10.0])

    assert result == [0.0, 0.5, 1.0]
