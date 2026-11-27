"""Deterministic unit tests for Tier-1 Regex Citation Verifier."""

from __future__ import annotations

from src.agent.nodes import verify_citations_regex


def test_verify_citations_regex_passes_valid_citations() -> None:
    chunks = [
        {"arxiv_id": "2310.11511", "page_number": 4, "text": "Self-RAG uses reflection tokens."},
        {"arxiv_id": "2401.15884", "page_number": 2, "text": "CRAG triggers Correct/Incorrect."},
    ]
    draft = (
        "Self-RAG uses reflection tokens [arXiv:2310.11511, p. 4], while CRAG adds "
        "a retrieval evaluator [arXiv:2401.15884, p. 2] and leads PopQA [SQL: benchmarks_reported]."
    )
    ok, errors = verify_citations_regex(draft, chunks, sql_results=[{"arxiv_id": "2401.15884"}])
    assert ok is True
    assert errors == []


def test_verify_citations_regex_catches_hallucinated_paper_id() -> None:
    chunks = [{"arxiv_id": "2310.11511", "page_number": 4, "text": "Self-RAG passage."}]
    draft = "According to a fabricated paper [arXiv:9999.12345, p. 1], accuracy is 99%."
    ok, errors = verify_citations_regex(draft, chunks)
    assert ok is False
    assert len(errors) == 1
    assert "9999.12345" in errors[0]
