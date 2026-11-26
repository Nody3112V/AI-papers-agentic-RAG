"""Fetch AI research papers from arXiv and populate papers.db (intended for Kaggle NB 1)."""

from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.init_db import init_papers_db
from src.core.config import get_settings

# Landmark papers per cluster so the corpus always includes well-known papers with rich benchmarks.
SEED_ARXIV_IDS: dict[str, list[str]] = {
    "rag": [
        "2310.11511",  # Self-RAG
        "2401.15884",  # Corrective RAG (CRAG)
        "2312.10997",  # Retrieval-Augmented Generation for LLMs: A Survey
        "2404.16130",  # GraphRAG (From Local to Global)
        "2402.03367",  # RAPTOR
    ],
    "agents": [
        "2210.03629",  # ReAct
        "2302.04761",  # Toolformer
        "2303.11366",  # Reflexion
        "2308.08155",  # AutoGen
        "2310.08560",  # MemGPT
    ],
    "finetuning": [
        "2305.14314",  # QLoRA
        "2305.18290",  # DPO
        "2402.09353",  # DoRA
        "2306.00978",  # AWQ
        "2403.07691",  # ORPO
    ],
    "evaluation": [
        "2309.15217",  # RAGAS
        "2303.16634",  # G-Eval
        "2306.05685",  # Judging LLM-as-a-Judge (MT-Bench)
        "2305.14251",  # FActScore
        "2305.11747",  # HaluEval
    ],
}

TOPIC_QUERIES: dict[str, str] = {
    "rag": '(cat:cs.CL OR cat:cs.IR) AND ("retrieval-augmented generation" OR "hybrid retrieval" OR "reranking")',
    "agents": '(cat:cs.AI OR cat:cs.CL) AND ("LLM agent" OR "tool use" OR "function calling" OR "planning")',
    "finetuning": '(cat:cs.LG OR cat:cs.CL) AND ("LoRA" OR "QLoRA" OR "direct preference optimization" OR "quantization")',
    "evaluation": '(cat:cs.CL OR cat:cs.AI) AND ("LLM-as-a-judge" OR "hallucination detection" OR "faithfulness")',
}

# Verified benchmark entries for landmark papers (gives accurate SQL/Hybrid queries out of the box).
SEED_BENCHMARKS: list[tuple[str, str, str, str, float, str]] = [
    ("2310.11511", "Self-RAG 7B", "PopQA", "Accuracy", 54.9, "Short-form QA"),
    ("2310.11511", "Self-RAG 13B", "PopQA", "Accuracy", 55.8, "Short-form QA"),
    ("2310.11511", "Self-RAG 13B", "TriviaQA", "Accuracy", 69.3, "Open-domain QA"),
    ("2401.15884", "CRAG + Self-RAG 7B", "PopQA", "Accuracy", 59.8, "With web fallback"),
    ("2401.15884", "CRAG 7B", "PopQA", "Accuracy", 54.5, "Corrective retrieval"),
    ("2402.03367", "RAPTOR + GPT-4", "QuALITY", "Accuracy", 82.6, "Long-document QA"),
    ("2210.03629", "ReAct (PaLM-540B)", "HotpotQA", "EM", 27.4, "Multi-hop reasoning"),
    ("2303.11366", "Reflexion + GPT-4", "HumanEval", "Pass@1", 91.0, "Code generation"),
    ("2305.14314", "Guanaco 65B (QLoRA)", "MMLU", "Accuracy", 62.7, "4-bit NormalFloat"),
    ("2305.14314", "Guanaco 33B (QLoRA)", "MMLU", "Accuracy", 57.8, "4-bit NormalFloat"),
    ("2402.09353", "DoRA (LLaMA-7B)", "Commonsense", "Accuracy", 78.1, "Weight-decomposed LoRA"),
    ("2306.05685", "GPT-4", "MT-Bench", "Score", 8.99, "LLM-as-a-Judge"),
    ("2305.11747", "GPT-3.5", "HaluEval", "Accuracy", 62.59, "QA hallucination detection"),
]

BENCHMARK_REGEX = re.compile(
    r"\b(MMLU|GSM8K|PopQA|HotpotQA|BEIR|TriviaQA|HumanEval|MT-Bench|TruthfulQA|HaluEval)\b"
    r"[^0-9]{0,25}(\d{2}(?:\.\d{1,2})?)\s*%?",
    re.IGNORECASE,
)

CANONICAL_BENCHMARKS = {
    "mmlu": "MMLU",
    "gsm8k": "GSM8K",
    "popqa": "PopQA",
    "hotpotqa": "HotpotQA",
    "beir": "BEIR",
    "triviaqa": "TriviaQA",
    "humaneval": "HumanEval",
    "mt-bench": "MT-Bench",
    "truthfulqa": "TruthfulQA",
    "halueval": "HaluEval",
}


def normalize_arxiv_id(raw_id: str) -> str:
    """Strip URL prefix and version suffix (e.g. '2310.11511v1' -> '2310.11511')."""
    short = raw_id.rsplit("/", maxsplit=1)[-1]
    return re.sub(r"v\d+$", "", short)


def detect_venue(comment: str | None) -> str:
    if not comment:
        return "arXiv"
    match = re.search(
        r"\b(ACL|EMNLP|NAACL|NeurIPS|ICLR|ICML|AAAI|SIGIR|KDD|CVPR)\s*20\d{2}\b",
        comment,
        re.IGNORECASE,
    )
    return match.group(0) if match else "arXiv"


def upsert_paper(
    conn: sqlite3.Connection,
    *,
    arxiv_id: str,
    title: str,
    abstract: str,
    published_date: str,
    year: int,
    primary_category: str,
    topic_cluster: str,
    venue: str,
    pdf_path: str,
    authors: list[str],
) -> None:
    conn.execute(
        """
        INSERT INTO papers (
            arxiv_id, title, abstract, published_date, year,
            primary_category, topic_cluster, venue, page_count, pdf_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?)
        ON CONFLICT(arxiv_id) DO UPDATE SET
            title = excluded.title,
            abstract = excluded.abstract,
            venue = excluded.venue,
            pdf_path = excluded.pdf_path
        """,
        (
            arxiv_id,
            title,
            abstract,
            published_date,
            year,
            primary_category,
            topic_cluster,
            venue,
            pdf_path,
        ),
    )

    for order, author_name in enumerate(authors, start=1):
        clean_name = author_name.strip()
        if not clean_name:
            continue
        conn.execute("INSERT OR IGNORE INTO authors (full_name) VALUES (?)", (clean_name,))
        author_id = conn.execute(
            "SELECT author_id FROM authors WHERE full_name = ?", (clean_name,)
        ).fetchone()[0]
        conn.execute(
            """
            INSERT OR IGNORE INTO paper_authors (arxiv_id, author_id, author_order)
            VALUES (?, ?, ?)
            """,
            (arxiv_id, author_id, order),
        )

    # Extract lightweight benchmark mentions from abstract if not already present.
    for match in BENCHMARK_REGEX.finditer(abstract):
        bench_key = match.group(1).lower()
        bench_name = CANONICAL_BENCHMARKS.get(bench_key, match.group(1))
        score = float(match.group(2))
        if score <= 0 or score > 100:
            continue
        exists = conn.execute(
            """
            SELECT 1 FROM benchmarks_reported
            WHERE arxiv_id = ? AND benchmark_name = ? AND ABS(score - ?) < 0.01
            """,
            (arxiv_id, bench_name, score),
        ).fetchone()
        if not exists:
            conn.execute(
                """
                INSERT INTO benchmarks_reported (
                    arxiv_id, model_name, benchmark_name, metric_name, score, notes
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (arxiv_id, title[:48], bench_name, "Score", score, "Extracted from abstract"),
            )


def seed_verified_benchmarks(conn: sqlite3.Connection) -> None:
    existing_papers = {
        row[0] for row in conn.execute("SELECT arxiv_id FROM papers").fetchall()
    }
    for arxiv_id, model_name, bench, metric, score, notes in SEED_BENCHMARKS:
        if arxiv_id not in existing_papers:
            continue
        exists = conn.execute(
            """
            SELECT 1 FROM benchmarks_reported
            WHERE arxiv_id = ? AND model_name = ? AND benchmark_name = ?
            """,
            (arxiv_id, model_name, bench),
        ).fetchone()
        if not exists:
            conn.execute(
                """
                INSERT INTO benchmarks_reported (
                    arxiv_id, model_name, benchmark_name, metric_name, score, notes
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (arxiv_id, model_name, bench, metric, score, notes),
            )


def fetch_papers(
    *,
    db_path: Path,
    pdf_dir: Path,
    limit: int = 60,
    min_year: int = 2022,
    skip_download: bool = False,
) -> None:
    try:
        import arxiv
    except ImportError as exc:
        raise RuntimeError(
            "Package 'arxiv' is not installed. Run this script on Kaggle Notebook 1 "
            "(or install group 'kaggle')."
        ) from exc

    init_papers_db(db_path)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    client = arxiv.Client(page_size=50, delay_seconds=3.0, num_retries=3)
    per_cluster = max(1, limit // len(TOPIC_QUERIES))
    seen_ids: set[str] = set()
    total_saved = 0

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA foreign_keys = ON;")

    try:
        for cluster, query in TOPIC_QUERIES.items():
            cluster_count = 0
            print(f"\n=== Fetching cluster '{cluster}' (target={per_cluster}) ===")

            # 1. Fetch curated landmark papers first
            seed_ids = SEED_ARXIV_IDS.get(cluster, [])
            if seed_ids:
                seed_search = arxiv.Search(id_list=seed_ids)
                for result in client.results(seed_search):
                    if cluster_count >= per_cluster:
                        break
                    aid = normalize_arxiv_id(result.entry_id)
                    if aid in seen_ids:
                        continue
                    saved = _process_result(
                        conn, result, aid, cluster, pdf_dir, skip_download=skip_download
                    )
                    if saved:
                        seen_ids.add(aid)
                        cluster_count += 1
                        total_saved += 1

            # 2. Fill remaining slots via topic query (2022-2026)
            if cluster_count < per_cluster:
                search = arxiv.Search(
                    query=query,
                    max_results=per_cluster * 4,
                    sort_by=arxiv.SortCriterion.Relevance,
                )
                for result in client.results(search):
                    if cluster_count >= per_cluster:
                        break
                    aid = normalize_arxiv_id(result.entry_id)
                    if aid in seen_ids:
                        continue
                    if result.published.year < min_year:
                        continue
                    saved = _process_result(
                        conn, result, aid, cluster, pdf_dir, skip_download=skip_download
                    )
                    if saved:
                        seen_ids.add(aid)
                        cluster_count += 1
                        total_saved += 1

            conn.commit()
            print(f"[CLUSTER DONE] '{cluster}': {cluster_count} papers")

        seed_verified_benchmarks(conn)
        conn.commit()

        bench_count = conn.execute("SELECT COUNT(*) FROM benchmarks_reported").fetchone()[0]
        author_count = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        print(
            f"\n[FETCH COMPLETE] papers={total_saved} | authors={author_count} | "
            f"benchmarks={bench_count} | db={db_path}"
        )
    finally:
        conn.close()


def _process_result(
    conn: sqlite3.Connection,
    result: object,
    arxiv_id: str,
    topic_cluster: str,
    pdf_dir: Path,
    *,
    skip_download: bool,
) -> bool:
    pdf_file = pdf_dir / f"{arxiv_id}.pdf"
    if not skip_download and not pdf_file.exists():
        try:
            result.download_pdf(dirpath=str(pdf_dir), filename=pdf_file.name)
            time.sleep(0.5)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"  [WARN] Failed to download {arxiv_id}: {exc}")
            return False

    title = " ".join(str(result.title).split())
    abstract = " ".join(str(result.summary).split())
    published_date = result.published.strftime("%Y-%m-%d")
    year = int(result.published.year)
    primary_category = str(result.primary_category)
    venue = detect_venue(getattr(result, "comment", None))
    authors = [a.name for a in result.authors]

    upsert_paper(
        conn,
        arxiv_id=arxiv_id,
        title=title,
        abstract=abstract,
        published_date=published_date,
        year=year,
        primary_category=primary_category,
        topic_cluster=topic_cluster,
        venue=venue,
        pdf_path=str(pdf_file),
        authors=authors,
    )
    print(f"  + [{topic_cluster}] {arxiv_id} ({year}) {title[:70]}")
    return True


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Fetch arXiv AI papers into papers.db.")
    parser.add_argument("--limit", type=int, default=60, help="Total papers to fetch (default: 60).")
    parser.add_argument("--min-year", type=int, default=2022, help="Minimum publication year.")
    parser.add_argument(
        "--db-path",
        default=settings.env.papers_db_path,
        help="Path to papers.db.",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/raw_papers",
        help="Directory to store downloaded PDFs.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Populate metadata only without downloading PDFs.",
    )
    args = parser.parse_args()
    fetch_papers(
        db_path=Path(args.db_path),
        pdf_dir=Path(args.pdf_dir),
        limit=args.limit,
        min_year=args.min_year,
        skip_download=args.skip_download,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
