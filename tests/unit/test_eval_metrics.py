"""Deterministic unit tests for Dynamic Evidence Matching and Recall@k / MRR / nDCG@k."""

from __future__ import annotations

import math

import pytest

from eval.eval_retrieval import evaluate_single_query, quote_matches_chunk


def test_quote_matches_chunk_exact_and_token_overlap() -> None:
    chunk = "Our framework uses reflection tokens (Retrieve, ISREL, ISSUP, ISUSE) for adaptive RAG."
    assert quote_matches_chunk("uses reflection tokens", chunk) is True
    # 5 out of 6 tokens overlap (~0.83 >= 0.75)
    assert quote_matches_chunk("framework uses reflection tokens for RAG", chunk) is True
    # Low overlap should not match
    assert quote_matches_chunk("quantum teleportation on mars colonies", chunk) is False


def test_evaluate_single_query_perfect_and_partial() -> None:
    retrieved = [
        {"arxiv_id": "9999.00001", "text": "Unrelated passage."},
        {"arxiv_id": "2310.11511", "text": "Four types of reflection tokens used in SELF-RAG."},
        {"arxiv_id": "2401.15884", "text": "CRAG triggers Correct, Incorrect, or Ambiguous actions."},
    ]
    res = evaluate_single_query(
        retrieved,
        expected_arxiv_ids=["2310.11511", "2401.15884"],
        evidence_quotes=[
            "Four types of reflection tokens used in SELF-RAG",
            "CRAG triggers Correct, Incorrect, or Ambiguous actions",
        ],
        top_k=3,
    )
    assert res["recall"] == pytest.approx(1.0)
    assert res["mrr"] == pytest.approx(0.5)  # First relevant hit is at rank 2
    expected_dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
    expected_idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert res["ndcg"] == pytest.approx(expected_dcg / expected_idcg)
