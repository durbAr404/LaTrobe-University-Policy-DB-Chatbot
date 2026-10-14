"""ValidationService grounding/confidence tests."""

import pytest

from app.rag.models import RetrievalResult
from app.services.validation_service import ValidationService


def make_result(
    chunk_id: str,
    vector_score: float = 0.5,
    lexical_score: float = 0.5,
    fused_score: float = 0.5,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        policy_id="p1",
        title="T",
        section="S",
        text="t",
        url="https://example.test",
        vector_score=vector_score,
        lexical_score=lexical_score,
        fused_score=fused_score,
    )


@pytest.fixture
def service():
    return ValidationService()


# ────────────────────────────────────────────────────────────────────────
# Citation-free answers
# ────────────────────────────────────────────────────────────────────────


def test_citation_free_answer_with_weak_evidence_is_accepted(service):
    """No citations + weak retrieval → treated as a valid conversational reply."""
    retrieved = [make_result("c1", fused_score=0.01)]

    is_valid, confidence, ids = service.validate("Hi, how can I help?", retrieved)

    assert is_valid is True
    assert confidence == 0.5
    assert ids == []


def test_citation_free_answer_with_no_retrieval_is_accepted(service):
    """No citations + nothing retrieved at all → still accepted."""
    is_valid, confidence, ids = service.validate("Hello there!", [])

    assert is_valid is True
    assert confidence == 0.5
    assert ids == []


def test_citation_free_answer_with_strong_evidence_is_grounding_failure(service):
    """No citations but retrieval was strong → the LLM should have cited something."""
    retrieved = [make_result("c1", fused_score=0.9)]

    is_valid, confidence, ids = service.validate("Some answer with no citation.", retrieved)

    assert is_valid is False
    assert confidence == 0.0
    assert ids == []


# ────────────────────────────────────────────────────────────────────────
# Cited answers
# ────────────────────────────────────────────────────────────────────────


def test_cited_answer_with_invalid_citation_is_invalid(service):
    retrieved = [make_result("c1")]

    is_valid, confidence, ids = service.validate("Fact [C99].", retrieved)

    assert is_valid is False
    assert confidence == 0.0
    assert ids == ["C99"]


def test_cited_answer_meeting_support_threshold_is_valid(service):
    """
    One valid citation against three results with full vector+lexical
    agreement: confidence = 0.50 + 0.10*1 + 0.25 = 0.85 >= min_generation_support.
    """
    retrieved = [make_result(f"c{i}") for i in range(3)]

    is_valid, confidence, ids = service.validate("Fact [C1].", retrieved)

    assert is_valid is True
    assert confidence == pytest.approx(0.85)
    assert ids == ["C1"]


def test_cited_answer_at_minimum_confidence_is_still_valid(service):
    """
    The lowest achievable confidence for ANY cited answer is 0.50 + 0.10*1 +
    0.0 = 0.60 (one citation, zero evidence agreement). That equals the
    default ``min_generation_support``, so it must pass (>=, not >).
    """
    retrieved = [make_result("c1", vector_score=0.0, lexical_score=0.0)]

    is_valid, confidence, ids = service.validate("Fact [C1].", retrieved)

    assert confidence == pytest.approx(0.60)
    assert is_valid is True


def test_confidence_increases_with_more_valid_citations(service):
    retrieved = [make_result(f"c{i}") for i in range(4)]

    _, confidence_one, _ = service.validate("Fact [C1].", retrieved)
    _, confidence_two, _ = service.validate("Fact [C1] and [C2].", retrieved)

    assert confidence_two > confidence_one


def test_confidence_is_capped_at_one(service):
    retrieved = [make_result(f"c{i}") for i in range(10)]
    answer = " ".join(f"[C{i + 1}]" for i in range(10))

    _, confidence, _ = service.validate(answer, retrieved)

    assert confidence == 1.0


def test_duplicate_citations_are_not_double_counted_beyond_extraction(service):
    """Citation IDs are de-duplicated before counting (see extract_citation_ids)."""
    retrieved = [make_result("c1", vector_score=0.0, lexical_score=0.0)]

    _, confidence, ids = service.validate("[C1] [C1] [C1]", retrieved)

    assert ids == ["C1"]
    assert confidence == pytest.approx(0.60)


# ────────────────────────────────────────────────────────────────────────
# _evidence_quality
# ────────────────────────────────────────────────────────────────────────


def test_evidence_quality_is_zero_with_no_results():
    assert ValidationService._evidence_quality([]) == 0.0


def test_evidence_quality_is_full_when_all_signals_present():
    retrieved = [make_result(f"c{i}", vector_score=0.5, lexical_score=0.5) for i in range(3)]

    assert ValidationService._evidence_quality(retrieved) == pytest.approx(0.25)


def test_evidence_quality_is_partial_with_vector_only():
    retrieved = [make_result(f"c{i}", vector_score=0.5, lexical_score=0.0) for i in range(3)]

    assert ValidationService._evidence_quality(retrieved) == pytest.approx(0.125)


def test_evidence_quality_only_considers_top_three():
    """A weak 4th+ result shouldn't drag down quality computed from the top 3."""
    strong_top3 = [make_result(f"c{i}", vector_score=0.5, lexical_score=0.5) for i in range(3)]
    weak_rest = [make_result("weak", vector_score=0.0, lexical_score=0.0)]

    quality_with_tail = ValidationService._evidence_quality(strong_top3 + weak_rest)
    quality_without_tail = ValidationService._evidence_quality(strong_top3)

    assert quality_with_tail == quality_without_tail


# ────────────────────────────────────────────────────────────────────────
# needs_escalation
# ────────────────────────────────────────────────────────────────────────


def test_needs_escalation_true_for_low_confidence():
    assert ValidationService.needs_escalation("A clear factual answer.", 0.5) is True


def test_needs_escalation_false_for_high_confidence_plain_answer():
    assert ValidationService.needs_escalation("A clear factual answer.", 0.9) is False


@pytest.mark.parametrize(
    "phrase",
    [
        "please contact the registrar",
        "seek official clarification",
        "ask the policy owner",
        "refer to the administration",
        "we cannot determine this",
        "this is not specified in policy",
        "there is insufficient information available",
    ],
)
def test_needs_escalation_true_for_escalation_phrases_even_with_high_confidence(phrase):
    assert ValidationService.needs_escalation(phrase, 0.95) is True


def test_needs_escalation_is_case_insensitive():
    assert ValidationService.needs_escalation("CONTACT the office.", 0.95) is True


# ────────────────────────────────────────────────────────────────────────
# disclaimer
# ────────────────────────────────────────────────────────────────────────


def test_disclaimer_is_nonempty_and_mentions_la_trobe():
    text = ValidationService.disclaimer()

    assert isinstance(text, str)
    assert "La Trobe" in text
