"""FastEmbed ONNX embedding manager (CPU on Laptop Zero-Config, CUDA fallback on Kaggle)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastembed import SparseTextEmbedding, TextEmbedding

from src.core.config import get_settings


def _resolve_onnx_providers(device: str) -> list[str] | None:
    if device != "cuda":
        return None
    import onnxruntime as ort

    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


@lru_cache(maxsize=2)
def get_dense_embedder(
    model_name: str | None = None,
    cache_dir: str | None = None,
    device: str = "cpu",
) -> TextEmbedding:
    settings = get_settings()
    name = model_name or settings.retrieval.dense_model
    cache = cache_dir or settings.retrieval.cache_dir
    Path(cache).mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=name,
        cache_dir=cache,
        providers=_resolve_onnx_providers(device),
    )


@lru_cache(maxsize=2)
def get_sparse_embedder(
    model_name: str | None = None,
    cache_dir: str | None = None,
) -> SparseTextEmbedding:
    settings = get_settings()
    name = model_name or settings.retrieval.sparse_model
    cache = cache_dir or settings.retrieval.cache_dir
    Path(cache).mkdir(parents=True, exist_ok=True)
    return SparseTextEmbedding(model_name=name, cache_dir=cache)


def embed_query_dense(query: str, *, device: str = "cpu") -> list[float]:
    """Embed a search query using BGE query prefix automatically via query_embed()."""
    model = get_dense_embedder(device=device)
    vec = next(iter(model.query_embed([query])))
    return vec.tolist()


def embed_query_sparse(query: str) -> tuple[list[int], list[float]]:
    """Embed a search query into BM25 sparse indices and values."""
    model = get_sparse_embedder()
    vec = next(iter(model.embed([query])))
    return vec.indices.tolist(), vec.values.tolist()
