"""Retrieval evaluation & 5-config Ablation Study (Recall@k, MRR@k, nDCG@k, Latency)."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import BaseReranker, get_reranker

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    normalized = html.unescape(text).lower()
    return set(TOKEN_RE.findall(normalized))


def quote_matches_chunk(
    quote: str,
    chunk_text: str,
    *,
    overlap_threshold: float = 0.75,
) -> bool:
    """Dynamic Evidence Matching: exact normalized substring OR token overlap >= 0.75."""
    norm_quote = " ".join(html.unescape(quote).lower().split())
    norm_chunk = " ".join(html.unescape(chunk_text).lower().split())
    if not norm_quote:
        return False
    if norm_quote in norm_chunk:
        return True
    q_tokens = tokenize(norm_quote)
    if not q_tokens:
        return False
    c_tokens = tokenize(norm_chunk)
    overlap = len(q_tokens & c_tokens) / len(q_tokens)
    return overlap >= overlap_threshold


def evaluate_single_query(
    retrieved: list[dict[str, Any]],
    *,
    expected_arxiv_ids: list[str],
    evidence_quotes: list[str],
    top_k: int = 5,
    overlap_threshold: float = 0.75,
) -> dict[str, float]:
    """Compute Recall@k, MRR@k, and nDCG@k for one query using Dynamic Evidence Matching."""
    top_chunks = retrieved[:top_k]
    expected_set = set(expected_arxiv_ids)

    matched_quotes: set[int] = set()
    relevances: list[int] = []
    first_rel_rank: int | None = None

    for rank, chunk in enumerate(top_chunks, start=1):
        aid = str(chunk.get("arxiv_id", ""))
        text = str(chunk.get("text", ""))
        is_rel = False

        if aid in expected_set:
            if evidence_quotes:
                for q_idx, quote in enumerate(evidence_quotes):
                    if quote_matches_chunk(quote, text, overlap_threshold=overlap_threshold):
                        matched_quotes.add(q_idx)
                        is_rel = True
            else:
                is_rel = True

        relevances.append(1 if is_rel else 0)
        if is_rel and first_rel_rank is None:
            first_rel_rank = rank

    # Recall@k: fraction of required evidence quotes (or expected papers) covered in top-k
    if evidence_quotes:
        recall = len(matched_quotes) / len(evidence_quotes)
        ideal_rel_count = min(top_k, len(evidence_quotes))
    else:
        matched_ids = {str(c.get("arxiv_id", "")) for c in top_chunks} & expected_set
        recall = len(matched_ids) / max(1, len(expected_set))
        ideal_rel_count = min(top_k, len(expected_set))

    mrr = (1.0 / first_rel_rank) if first_rel_rank is not None else 0.0

    dcg = sum((2**rel - 1) / math.log2(rank + 1) for rank, rel in enumerate(relevances, start=1))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_rel_count + 1))
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return {"recall": recall, "mrr": mrr, "ndcg": ndcg}


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip() or "local"
    except (OSError, subprocess.SubprocessError):
        return "local"


def _build_ablation_configs(device: str) -> dict[str, dict[str, Any]]:
    return {
        "dense_only": {
            "label": "1. Dense Only (bge-small-en-v1.5)",
            "mode": "dense_only",
            "reranker_backend": None,
        },
        "sparse_only": {
            "label": "2. Sparse Only (Qdrant/bm25)",
            "mode": "sparse_only",
            "reranker_backend": None,
        },
        "hybrid_rrf": {
            "label": "3. Hybrid + RRF (k=60)",
            "mode": "hybrid_rrf",
            "reranker_backend": None,
        },
        "hybrid_rerank_onnx": {
            "label": "4. Hybrid + RRF + ONNX Reranker (Laptop)",
            "mode": "hybrid_rerank",
            "reranker_backend": "fastembed",
        },
        "hybrid_rerank_bge_gpu": {
            "label": "5. Hybrid + RRF + BGE Reranker (Kaggle GPU)",
            "mode": "hybrid_rerank",
            "reranker_backend": "sentence_transformers",
            "reranker_device": device,
        },
    }


def format_markdown_table(summary_rows: list[dict[str, Any]], top_k: int) -> str:
    lines = [
        f"| Cấu hình thực nghiệm | Recall@{top_k} | MRR@{top_k} | nDCG@{top_k} | Độ trễ TB (ms) |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['label']} | {row['recall_at_k']:.2%} | "
            f"{row['mrr_at_k']:.4f} | {row['ndcg_at_k']:.4f} | {row['avg_latency_ms']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def run_ablation(
    *,
    dataset_path: Path,
    selected_configs: list[str],
    top_k: int = 5,
    device: str = "cpu",
    out_dir: Path = PROJECT_ROOT / "eval" / "reports",
) -> dict[str, Any]:
    raw_ds = json.loads(dataset_path.read_text(encoding="utf-8"))
    rag_items = [
        item for item in raw_ds if item.get("category") in {"factual_rag", "multi_hop_rag"}
    ]
    print(f"[EVAL RETRIEVAL] Loaded {len(rag_items)} RAG queries (Factual + Multi-hop), top_k={top_k}")

    all_configs = _build_ablation_configs(device)
    searcher = HybridSearcher(device=device)
    summary_rows: list[dict[str, Any]] = []
    per_config_details: dict[str, list[dict[str, Any]]] = {}

    for cfg_key in selected_configs:
        if cfg_key not in all_configs:
            raise ValueError(f"Unknown ablation config: {cfg_key}")
        cfg = all_configs[cfg_key]
        print(f"\n--- Running: {cfg['label']} ---")

        reranker: BaseReranker | None = None
        if cfg["reranker_backend"] == "fastembed":
            reranker = get_reranker(backend="fastembed")
        elif cfg["reranker_backend"] == "sentence_transformers":
            reranker = get_reranker(
                backend="sentence_transformers",
                device=cfg.get("reranker_device", device),
            )

        recalls: list[float] = []
        mrrs: list[float] = []
        ndcgs: list[float] = []
        latencies: list[float] = []
        details: list[dict[str, Any]] = []

        for item in rag_items:
            t0 = time.perf_counter()
            hits = searcher.search(
                item["question"],
                mode=cfg["mode"],
                top_k=top_k,
                reranker=reranker,
                min_rerank_score=-1e9,  # Keep full top-k for ranking ablation fairness
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            scores = evaluate_single_query(
                hits,
                expected_arxiv_ids=item["expected_arxiv_ids"],
                evidence_quotes=item["evidence_quotes"],
                top_k=top_k,
            )
            recalls.append(scores["recall"])
            mrrs.append(scores["mrr"])
            ndcgs.append(scores["ndcg"])
            latencies.append(elapsed_ms)
            details.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    **scores,
                    "latency_ms": round(elapsed_ms, 2),
                    "top_arxiv_ids": [h["arxiv_id"] for h in hits],
                }
            )

        n = max(1, len(rag_items))
        row = {
            "config": cfg_key,
            "label": cfg["label"],
            "num_queries": len(rag_items),
            "recall_at_k": round(sum(recalls) / n, 4),
            "mrr_at_k": round(sum(mrrs) / n, 4),
            "ndcg_at_k": round(sum(ndcgs) / n, 4),
            "avg_latency_ms": round(sum(latencies) / n, 1),
        }
        summary_rows.append(row)
        per_config_details[cfg_key] = details
        print(
            f"  Recall@{top_k}={row['recall_at_k']:.2%} | MRR@{top_k}={row['mrr_at_k']:.4f} | "
            f"nDCG@{top_k}={row['ndcg_at_k']:.4f} | avg={row['avg_latency_ms']:.1f} ms"
        )

    sha = _git_short_sha()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "git_commit": sha,
        "top_k": top_k,
        "num_queries": len(rag_items),
        "summary": summary_rows,
        "details": per_config_details,
    }
    json_path = out_dir / f"retrieval_ablation_{sha}.json"
    md_path = out_dir / f"retrieval_ablation_{sha}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_table = format_markdown_table(summary_rows, top_k=top_k)
    md_path.write_text(f"# Retrieval Ablation Study (commit `{sha}`)\n\n{md_table}", encoding="utf-8")

    print(f"\n=== ABLATION SUMMARY TABLE ===\n{md_table}")
    print(f"[SAVED] {json_path}\n[SAVED] {md_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run 5-config retrieval ablation study.")
    parser.add_argument("--dataset", default="data/eval/golden_dataset.json")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--configs",
        default="all",
        help="Comma-separated config keys or 'all' (5 configs).",
    )
    parser.add_argument("--out-dir", default="eval/reports")
    args = parser.parse_args()

    all_keys = [
        "dense_only",
        "sparse_only",
        "hybrid_rrf",
        "hybrid_rerank_onnx",
        "hybrid_rerank_bge_gpu",
    ]
    selected = all_keys if args.configs == "all" else [c.strip() for c in args.configs.split(",") if c.strip()]

    run_ablation(
        dataset_path=Path(args.dataset),
        selected_configs=selected,
        top_k=args.top_k,
        device=args.device,
        out_dir=Path(args.out_dir),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
