"""Deterministic unit tests for FastMCP server Resources, Tools, Prompt, and JSONL audit log."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.init_db import init_papers_db
from src.core.config import get_settings
from src.mcp_server import server as mcp_server


@pytest.fixture()
def isolated_mcp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    db_path = tmp_path / "papers.db"
    audit_path = tmp_path / "mcp_audit.jsonl"
    init_papers_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "INSERT INTO papers (arxiv_id, title, abstract, published_date, year, "
        "primary_category, topic_cluster, venue, page_count, pdf_path) "
        "VALUES ('2310.11511', 'Self-RAG', 'Self-reflective RAG.', '2023-10-17', 2023, "
        "'cs.CL', 'rag', 'ICLR 2024', 30, '2310.11511.pdf')"
    )
    conn.execute("INSERT INTO authors (full_name) VALUES ('Akari Asai')")
    conn.execute("INSERT INTO paper_authors (arxiv_id, author_id, author_order) VALUES ('2310.11511', 1, 1)")
    conn.execute(
        "INSERT INTO benchmarks_reported (arxiv_id, model_name, benchmark_name, metric_name, score) "
        "VALUES ('2310.11511', 'Self-RAG 13B', 'PopQA', 'Accuracy', 55.8)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("PAPERS_DB_PATH", str(db_path))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(audit_path))
    get_settings.cache_clear()
    yield db_path, audit_path
    get_settings.cache_clear()


def test_mcp_resources_return_schema_catalog_and_stats(
    isolated_mcp_env: tuple[Path, Path],
) -> None:
    schema_text = mcp_server.resource_db_schema()
    assert "CREATE TABLE papers" in schema_text
    assert "CREATE TABLE reading_notes" in schema_text

    catalog_text = mcp_server.resource_papers_catalog()
    assert "2310.11511" in catalog_text
    assert "Self-RAG" in catalog_text

    stats = json.loads(mcp_server.resource_papers_stats())
    assert stats["total_papers"] == 1
    assert stats["benchmarks_available"] == ["PopQA"]


def test_mcp_query_papers_db_and_audit_log(isolated_mcp_env: tuple[Path, Path]) -> None:
    _, audit_path = isolated_mcp_env
    ok = mcp_server.query_papers_db("SELECT arxiv_id, title FROM papers")
    assert ok["status"] == "ok"
    assert ok["row_count"] == 1
    assert ok["rows"][0]["arxiv_id"] == "2310.11511"

    blocked = mcp_server.query_papers_db("DROP TABLE papers")
    assert blocked["status"] == "error"

    lines = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert lines[0]["status"] == "ok"
    assert lines[1]["status"] == "blocked"


def test_mcp_get_metadata_and_create_reading_note(isolated_mcp_env: tuple[Path, Path]) -> None:
    meta = mcp_server.get_paper_metadata("2310.11511")
    assert meta["status"] == "ok"
    assert meta["authors"] == ["Akari Asai"]
    assert meta["benchmarks"][0]["score"] == 55.8

    created = mcp_server.create_reading_note(
        arxiv_id="2310.11511",
        note="Study reflection tokens in Section 3.2.",
        tags=["rag", "must-read"],
        priority="high",
    )
    assert created["status"] == "ok"
    assert created["note_id"] == 1

    bad_paper = mcp_server.create_reading_note(arxiv_id="9999.99999", note="Invalid")
    assert bad_paper["status"] == "error"

    bad_priority = mcp_server.create_reading_note(
        arxiv_id="2310.11511", note="Bad priority", priority="urgent"
    )
    assert bad_priority["status"] == "error"


def test_mcp_prompt_compare_papers() -> None:
    prompt = mcp_server.compare_papers_methodology("2310.11511", "2401.15884")
    assert "arXiv:2310.11511" in prompt
    assert "arXiv:2401.15884" in prompt
