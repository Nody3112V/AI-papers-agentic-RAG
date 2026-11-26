"""Idempotent Qdrant indexer supporting Qdrant Cloud Free Tier and 0-Docker Local Path."""

from __future__ import annotations

import hashlib
import os
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from src.ingestion.chunker import PaperChunk


def make_chunk_point_id(arxiv_id: str, page_number: int, chunk_index: int, text: str) -> str:
    """Deterministic UUIDv5 point ID for 100% idempotent Qdrant upserts."""
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    key = f"{arxiv_id}:p{page_number}:c{chunk_index}:{content_hash}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))


def get_qdrant_client(
    *,
    mode: str | None = None,
    local_path: str | None = None,
) -> QdrantClient:
    effective_mode = mode or os.getenv("QDRANT_MODE", "cloud")
    if effective_mode == "cloud":
        return QdrantClient(
            url=os.environ["QDRANT_URL"],
            api_key=os.environ["QDRANT_API_KEY"],
            timeout=30.0,
        )
    path = local_path or os.getenv("QDRANT_LOCAL_PATH", "data/qdrant_local")
    return QdrantClient(path=path)


def ensure_collection(
    client: QdrantClient,
    collection_name: str = "ai_papers_chunks",
    dense_dim: int = 384,
) -> None:
    """Create hybrid dense + sparse BM25 collection with payload indexes if missing."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": models.VectorParams(
                    size=dense_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )
        for field, schema in [
            ("arxiv_id", models.PayloadSchemaType.KEYWORD),
            ("topic_cluster", models.PayloadSchemaType.KEYWORD),
            ("year", models.PayloadSchemaType.INTEGER),
            ("page_number", models.PayloadSchemaType.INTEGER),
            ("has_table", models.PayloadSchemaType.BOOL),
        ]:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field,
                field_schema=schema,
            )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[PaperChunk],
    *,
    dense_embedder: Any,
    sparse_embedder: Any,
    collection_name: str = "ai_papers_chunks",
    batch_size: int = 64,
) -> int:
    """Embed and upsert chunks in batches with deterministic UUIDv5 IDs."""
    if not chunks:
        return 0

    ensure_collection(client, collection_name=collection_name)
    total_upserted = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.embedding_text for c in batch]

        dense_vecs = list(dense_embedder.passage_embed(texts))
        sparse_vecs = list(sparse_embedder.embed(texts))

        points: list[models.PointStruct] = []
        for chunk, d_vec, s_vec in zip(batch, dense_vecs, sparse_vecs, strict=True):
            point_id = make_chunk_point_id(
                chunk.arxiv_id,
                chunk.page_number,
                chunk.chunk_index,
                chunk.text,
            )
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": d_vec.tolist(),
                        "sparse": models.SparseVector(
                            indices=s_vec.indices.tolist(),
                            values=s_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "arxiv_id": chunk.arxiv_id,
                        "title": chunk.title,
                        "year": chunk.year,
                        "topic_cluster": chunk.topic_cluster,
                        "page_number": chunk.page_number,
                        "section_path": chunk.section_path,
                        "chunk_index": chunk.chunk_index,
                        "has_table": chunk.has_table,
                        "text": chunk.text,
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)
        total_upserted += len(points)

    return total_upserted
