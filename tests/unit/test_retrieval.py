"""Deterministic unit tests for RRF math, Qdrant payload filters, and reranker filtering."""

from __future__ import annotations

import pytest

from src.retrieval.hybrid_search import build_qdrant_filter, reciprocal_rank_fusion
from src.retrieval.reranker import BaseReranker


class _FakeReranker(BaseReranker):
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        return self._scores[: len(documents)]


def test_reciprocal_rank_fusion_exact_formula() -> None:
    dense_list = ["doc_a", "doc_b", "doc_c"]
    sparse_list = ["doc_b", "doc_d", "doc_a"]
    fused = dict(reciprocal_rank_fusion([dense_list, sparse_list], k=60))

    # doc_b: rank 2 in dense + rank 1 in sparse
    assert fused["doc_b"] == pytest.approx(1 / 62 + 1 / 61)
    # doc_a: rank 1 in dense + rank 3 in sparse
    assert fused["doc_a"] == pytest.approx(1 / 61 + 1 / 63)
    # doc_b must outrank doc_a
    ordered_ids = [doc_id for doc_id, _ in reciprocal_rank_fusion([dense_list, sparse_list], k=60)]
    assert ordered_ids[0] == "doc_b"
    assert ordered_ids[1] == "doc_a"


def test_build_qdrant_filter_conditions() -> None:
    assert build_qdrant_filter() is None
    flt = build_qdrant_filter(
        arxiv_ids=["2310.11511", "2401.15884"],
        year_min=2024,
        only_tables=True,
    )
    assert flt is not None
    assert flt.must is not None
    assert len(flt.must) == 3
    keys = {cond.key for cond in flt.must}
    assert keys == {"arxiv_id", "year", "has_table"}


def test_reranker_sorts_and_filters_by_min_score() -> None:
    candidates = [
        {"point_id": "1", "text": "low relevance"},
        {"point_id": "2", "text": "high relevance"},
        {"point_id": "3", "text": "medium relevance"},
    ]
    reranker = _FakeReranker([0.05, 0.92, 0.41])
    out = reranker.rerank("self-rag reflection tokens", candidates, top_k=2, min_score=0.15)
    assert [c["point_id"] for c in out] == ["2", "3"]
    assert out[0]["rerank_score"] == pytest.approx(0.92)
    assert out[1]["rerank_score"] == pytest.approx(0.41)
