"""Run full ingestion pipeline: PDF -> Docling/PyMuPDF4LLM -> Chunk -> Qdrant Cloud."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastembed import SparseTextEmbedding, TextEmbedding

from src.core.config import get_settings
from src.ingestion.chunker import PaperChunk, chunk_markdown
from src.ingestion.indexer import get_qdrant_client, upsert_chunks
from src.ingestion.pdf_parser import parse_pdf_to_markdown


def load_embedders(
    *,
    dense_model: str,
    sparse_model: str,
    cache_dir: str,
    device: str = "cpu",
) -> tuple[TextEmbedding, SparseTextEmbedding]:
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else None
    dense = TextEmbedding(model_name=dense_model, cache_dir=cache_dir, providers=providers)
    sparse = SparseTextEmbedding(model_name=sparse_model, cache_dir=cache_dir)
    return dense, sparse


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Parse PDFs, chunk, and index into Qdrant.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--target", choices=["cloud", "local_path"], default=settings.effective_qdrant_mode)
    parser.add_argument("--parser", choices=["docling", "pymupdf4llm"], default=settings.ingestion.parser)
    parser.add_argument("--db-path", default=settings.env.papers_db_path)
    parser.add_argument("--markdown-dir", default="data/processed/markdown")
    parser.add_argument("--local-backup-path", default=settings.qdrant.local_path)
    parser.add_argument("--export-local-backup", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="0 = process all papers in DB.")
    parser.add_argument("--overwrite-md", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    md_dir = Path(args.markdown_dir)
    md_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT arxiv_id, title, year, topic_cluster, pdf_path FROM papers ORDER BY year DESC, arxiv_id ASC"
    ).fetchall()
    if args.limit > 0:
        rows = rows[: args.limit]

    print(f"[INGESTION] Papers in DB to process: {len(rows)} | parser={args.parser} | device={args.device}")

    all_chunks: list[PaperChunk] = []
    for idx, row in enumerate(rows, start=1):
        arxiv_id = row["arxiv_id"]
        pdf_path = Path(row["pdf_path"])
        md_path = md_dir / f"{arxiv_id}.md"

        if not pdf_path.exists() and not md_path.exists():
            print(f"  [SKIP] {arxiv_id}: neither PDF nor cached Markdown exists.")
            continue

        markdown, page_count = parse_pdf_to_markdown(
            pdf_path,
            md_path,
            parser=args.parser,
            do_ocr=settings.ingestion.docling_do_ocr,
            do_table_structure=settings.ingestion.docling_do_table_structure,
            exclude_sections=settings.ingestion.exclude_sections,
            overwrite=args.overwrite_md,
        )
        conn.execute("UPDATE papers SET page_count = ? WHERE arxiv_id = ?", (page_count, arxiv_id))

        paper_chunks = chunk_markdown(
            markdown,
            arxiv_id=arxiv_id,
            title=row["title"],
            year=int(row["year"]),
            topic_cluster=row["topic_cluster"],
            chunk_size_tokens=settings.ingestion.chunk_size_tokens,
            chunk_overlap_tokens=settings.ingestion.chunk_overlap_tokens,
            max_table_chunk_tokens=settings.ingestion.max_table_chunk_tokens,
            inject_contextual_header=settings.ingestion.inject_contextual_header,
        )
        all_chunks.extend(paper_chunks)
        table_chunks = sum(1 for c in paper_chunks if c.has_table)
        print(
            f"  [{idx}/{len(rows)}] {arxiv_id} | pages={page_count} | "
            f"chunks={len(paper_chunks)} (tables={table_chunks})"
        )

    conn.commit()
    conn.close()

    print(f"\n[EMBEDDING] Total chunks ready: {len(all_chunks)}")
    dense_embedder, sparse_embedder = load_embedders(
        dense_model=settings.retrieval.dense_model,
        sparse_model=settings.retrieval.sparse_model,
        cache_dir=settings.retrieval.cache_dir,
        device=args.device,
    )

    collection = settings.qdrant.collection_name
    client = get_qdrant_client(mode=args.target, local_path=args.local_backup_path)
    upserted = upsert_chunks(
        client,
        all_chunks,
        dense_embedder=dense_embedder,
        sparse_embedder=sparse_embedder,
        collection_name=collection,
        batch_size=args.batch_size,
    )
    info = client.get_collection(collection)
    print(
        f"[QDRANT {args.target.upper()}] upserted={upserted} | "
        f"total points_count={info.points_count}"
    )

    if args.export_local_backup and args.target == "cloud":
        local_client = get_qdrant_client(mode="local_path", local_path=args.local_backup_path)
        local_upserted = upsert_chunks(
            local_client,
            all_chunks,
            dense_embedder=dense_embedder,
            sparse_embedder=sparse_embedder,
            collection_name=collection,
            batch_size=args.batch_size,
        )
        print(f"[LOCAL BACKUP] upserted={local_upserted} -> {args.local_backup_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
