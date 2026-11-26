"""Deterministic unit tests for PDF markdown cleanup, table-aware chunking, and UUIDv5 IDs."""

from __future__ import annotations

from src.ingestion.chunker import chunk_markdown
from src.ingestion.indexer import make_chunk_point_id
from src.ingestion.pdf_parser import clean_academic_markdown


def test_clean_academic_markdown_strips_references_and_rejoins_hyphens() -> None:
    raw = (
        "<!-- PAGE: 1 -->\n"
        "# 1. Introduction\n"
        "001 Hybrid re-\ntrieval improves recall.\n\n"
        "## References\n"
        "[1] Some Paper 2024.\n"
    )
    cleaned = clean_academic_markdown(raw, exclude_sections=["references", "acknowledgments"])
    assert "Hybrid retrieval improves recall." in cleaned
    assert "001 " not in cleaned
    assert "Some Paper 2024" not in cleaned


def test_chunker_preserves_markdown_table_and_page_number() -> None:
    md = (
        "<!-- PAGE: 1 -->\n"
        "# 1. Method\n"
        "We propose a two-stage hybrid retrieval pipeline.\n\n"
        "<!-- PAGE: 2 -->\n"
        "## 2.1 Results\n"
        "Table 1: Benchmark comparison on PopQA.\n"
        "| Model | PopQA | TriviaQA |\n"
        "| --- | --- | --- |\n"
        "| Dense Only | 48.2 | 61.0 |\n"
        "| Hybrid + Rerank | 55.8 | 69.3 |\n"
    )
    chunks = chunk_markdown(
        md,
        arxiv_id="2310.11511",
        title="Self-RAG",
        year=2023,
        topic_cluster="rag",
    )
    table_chunks = [c for c in chunks if c.has_table]
    assert len(table_chunks) == 1
    assert table_chunks[0].page_number == 2
    assert "Table 1:" in table_chunks[0].text
    assert "| Hybrid + Rerank | 55.8 | 69.3 |" in table_chunks[0].text
    assert table_chunks[0].section_path == "1. Method > 2.1 Results"
    assert table_chunks[0].embedding_text.startswith("Paper: Self-RAG (2023)")


def test_chunker_splits_oversized_table_repeating_header() -> None:
    rows = "\n".join(f"| Model_{i} | {50 + i * 0.1:.1f} | notes about experiment {i} |" for i in range(40))
    md = (
        "<!-- PAGE: 4 -->\n"
        "# Experiments\n"
        "| Model | Score | Notes |\n"
        "| --- | --- | --- |\n"
        f"{rows}\n"
    )
    chunks = chunk_markdown(
        md,
        arxiv_id="2401.15884",
        title="CRAG",
        year=2024,
        topic_cluster="rag",
        max_table_chunk_tokens=120,
    )
    assert len(chunks) > 1
    for c in chunks:
        assert c.has_table is True
        assert "| Model | Score | Notes |" in c.text
        assert "| --- | --- | --- |" in c.text


def test_make_chunk_point_id_is_deterministic() -> None:
    id1 = make_chunk_point_id("2310.11511", 4, 2, "sample chunk text")
    id2 = make_chunk_point_id("2310.11511", 4, 2, "sample chunk text")
    id3 = make_chunk_point_id("2310.11511", 4, 3, "sample chunk text")
    assert id1 == id2
    assert id1 != id3
