"""3-stage Hybrid Retrieval: Qdrant Prefetch (Dense + BM25) + RRF (k=60) + Cross-Encoder Rerank."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient, models

from src.core.config import get_settings
from src.ingestion.indexer import get_qdrant_client
from src.retrieval.embeddings import embed_query_dense, embed_query_sparse
from src.retrieval.reranker import BaseReranker, get_reranker


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    *,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Compute Reciprocal Rank Fusion: RRF(d) = sum(1 / (k + rank_r(d))), 1-indexed ranks."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def build_qdrant_filter(
    *,
    arxiv_ids: list[str] | None = None,
    year_min: int | None = None,
    topic_cluster: str | None = None,
    only_tables: bool = False,
) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if arxiv_ids:
        conditions.append(
            models.FieldCondition(
                key="arxiv_id",
                match=models.MatchAny(any=arxiv_ids),
            )
        )
    if year_min is not None:
        conditions.append(
            models.FieldCondition(
                key="year",
                range=models.Range(gte=year_min),
            )
        )
    if topic_cluster:
        conditions.append(
            models.FieldCondition(
                key="topic_cluster",
                match=models.MatchValue(value=topic_cluster),
            )
        )
    if only_tables:
        conditions.append(
            models.FieldCondition(
                key="has_table",
                match=models.MatchValue(value=True),
            )
        )
    if not conditions:
        return None
    return models.Filter(must=conditions)


def _point_to_dict(point: Any) -> dict[str, Any]:
    payload = dict(point.payload or {})
    return {
        "point_id": str(point.id),
        "score": round(float(point.score), 4),
        "arxiv_id": payload.get("arxiv_id", ""),
        "title": payload.get("title", ""),
        "year": payload.get("year"),
        "topic_cluster": payload.get("topic_cluster", ""),
        "page_number": payload.get("page_number", 1),
        "section_path": payload.get("section_path", ""),
        "chunk_index": payload.get("chunk_index", 0),
        "has_table": bool(payload.get("has_table", False)),
        "text": payload.get("text", ""),
    }


class HybridSearcher:
    """Unified retriever supporting dense_only, sparse_only, hybrid_rrf, and hybrid_rerank."""

    def __init__(
        self,
        *,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
        device: str = "cpu",
    ) -> None:
        settings = get_settings()
        self.settings = settings
        self.client = client or get_qdrant_client(mode=settings.effective_qdrant_mode)
        self.collection_name = collection_name or settings.qdrant.collection_name
        self.device = device

    def search_dense(
        self,
        query: str,
        *,
        top_k: int = 5,
        qdrant_filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        dense_vec = embed_query_dense(query, device=self.device)
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_vec,
            using="dense",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [_point_to_dict(pt) for pt in res.points]

    def search_sparse(
        self,
        query: str,
        *,
        top_k: int = 5,
        qdrant_filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        indices, values = embed_query_sparse(query)
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=models.SparseVector(indices=indices, values=values),
            using="sparse",
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )
        return [_point_to_dict(pt) for pt in res.points]

    def search_hybrid_rrf(
        self,
        query: str,
        *,
        dense_top_k: int | None = None,
        sparse_top_k: int | None = None,
        limit: int | None = None,
        qdrant_filter: models.Filter | None = None,
    ) -> list[dict[str, Any]]:
        cfg = self.settings.retrieval
        d_k = dense_top_k or cfg.dense_top_k
        s_k = sparse_top_k or cfg.sparse_top_k
        out_k = limit or cfg.rrf_candidates

        dense_vec = embed_query_dense(query, device=self.device)
        indices, values = embed_query_sparse(query)

        res = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vec,
                    using="dense",
                    filter=qdrant_filter,
                    limit=d_k,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using="sparse",
                    filter=qdrant_filter,
                    limit=s_k,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=out_k,
            with_payload=True,
        )
        return [_point_to_dict(pt) for pt in res.points]

    def search(
        self,
        query: str,
        *,
        mode: str = "hybrid_rerank",
        arxiv_ids: list[str] | None = None,
        year_min: int | None = None,
        topic_cluster: str | None = None,
        only_tables: bool = False,
        top_k: int | None = None,
        reranker: BaseReranker | None = None,
        min_rerank_score: float | None = None,
    ) -> list[dict[str, Any]]:
        cfg = self.settings.retrieval
        k = top_k or cfg.rerank_top_k
        qfilter = build_qdrant_filter(
            arxiv_ids=arxiv_ids,
            year_min=year_min,
            topic_cluster=topic_cluster,
            only_tables=only_tables,
        )

        if mode == "dense_only":
            return self.search_dense(query, top_k=k, qdrant_filter=qfilter)
        if mode == "sparse_only":
            return self.search_sparse(query, top_k=k, qdrant_filter=qfilter)
        if mode == "hybrid_rrf":
            return self.search_hybrid_rrf(query, limit=k, qdrant_filter=qfilter)
        if mode == "hybrid_rerank":
            candidates = self.search_hybrid_rrf(
                query,
                limit=max(cfg.rrf_candidates, k * 3),
                qdrant_filter=qfilter,
            )
            active_reranker = reranker or get_reranker(backend="fastembed")
            threshold = (
                cfg.min_rerank_score if min_rerank_score is None else min_rerank_score
            )
            return active_reranker.rerank(
                query,
                candidates,
                top_k=k,
                min_score=threshold,
            )
        raise ValueError(f"Unsupported retrieval mode: {mode}")
