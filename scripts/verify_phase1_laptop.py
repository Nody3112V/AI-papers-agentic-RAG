"""Verify Phase 1 Definition of Done from Laptop: SQLite papers.db + Qdrant Cloud filters."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qdrant_client import models

from src.core.config import get_settings
from src.ingestion.indexer import get_qdrant_client


def main() -> int:
    settings = get_settings()
    db_path = Path(settings.env.papers_db_path)

    print("=== Phase 1 Laptop Verification ===")
    if not db_path.exists():
        print(f"[ERROR] Missing {db_path}. Download phase1_data_for_laptop.zip from Kaggle first.")
        return 1

    conn = sqlite3.connect(str(db_path))
    try:
        papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        authors = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        benchmarks = conn.execute("SELECT COUNT(*) FROM benchmarks_reported").fetchone()[0]
        clusters = conn.execute(
            "SELECT topic_cluster, COUNT(*) FROM papers GROUP BY topic_cluster ORDER BY topic_cluster"
        ).fetchall()
    finally:
        conn.close()

    print(f"SQLite papers.db : papers={papers} | authors={authors} | benchmarks={benchmarks}")
    print(f"Topic clusters   : {dict(clusters)}")

    client = get_qdrant_client(mode=settings.effective_qdrant_mode)
    collection = settings.qdrant.collection_name
    info = client.get_collection(collection)
    print(f"Qdrant Cloud     : collection={collection} | points_count={info.points_count}")

    # Test payload filter (year >= 2024 and has_table == True)
    table_points, _ = client.scroll(
        collection_name=collection,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(key="year", range=models.Range(gte=2024)),
                models.FieldCondition(key="has_table", match=models.MatchValue(value=True)),
            ]
        ),
        limit=3,
        with_payload=True,
    )
    print(f"Filtered tables (year>=2024, has_table=True): found {len(table_points)} sample chunks")
    for pt in table_points:
        payload = pt.payload or {}
        print(
            f"  - arXiv:{payload.get('arxiv_id')} (p.{payload.get('page_number')}) "
            f"| {payload.get('section_path')}"
        )

    if papers < 10 or (info.points_count or 0) < 100:
        print("[WARN] Corpus looks smaller than expected (target: 50-80 papers, 1500-3500 chunks).")
    else:
        print("=== PHASE 1 OK ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
