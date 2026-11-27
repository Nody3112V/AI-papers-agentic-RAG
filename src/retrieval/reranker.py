"""Cross-encoder reranker supporting lightweight FastEmbed ONNX (Laptop) and BGE GPU (Kaggle)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.core.config import get_settings


class BaseReranker:
    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        raise NotImplementedError

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        *,
        top_k: int = 5,
        min_score: float = 0.15,
        text_key: str = "text",
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        docs = [str(item.get(text_key, "")) for item in candidates]
        scores = self.score_pairs(query, docs)
        scored: list[dict[str, Any]] = []
        for item, score in zip(candidates, scores, strict=True):
            score_f = float(score)
            if score_f < min_score:
                continue
            enriched = dict(item)
            enriched["rerank_score"] = round(score_f, 4)
            scored.append(enriched)
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored[:top_k]


class FastEmbedReranker(BaseReranker):
    """Lightweight ONNX CPU cross-encoder (~80-150 MB RAM, 0 PyTorch)."""

    def __init__(self, model_name: str, cache_dir: str = "./.cache/fastembed") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._model = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._model.rerank(query, documents)]


class SentenceTransformerReranker(BaseReranker):
    """Heavy GPU cross-encoder (e.g. BAAI/bge-reranker-base) for Kaggle NB 2 ablation."""

    def __init__(self, model_name: str, device: str | None = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name, device=device)

    def score_pairs(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]


@lru_cache(maxsize=4)
def get_reranker(
    *,
    backend: str = "fastembed",
    model_name: str | None = None,
    device: str | None = None,
) -> BaseReranker:
    settings = get_settings()
    if backend == "fastembed":
        name = model_name or settings.retrieval.reranker_model_laptop
        return FastEmbedReranker(model_name=name, cache_dir=settings.retrieval.cache_dir)
    if backend == "sentence_transformers":
        name = model_name or settings.retrieval.reranker_model_kaggle_heavy
        return SentenceTransformerReranker(model_name=name, device=device)
    raise ValueError(f"Unsupported reranker backend: {backend}")
