"""FastMCP Server exposing 3 Resources, 4 Tools, and 1 Prompt over stdio or Streamable HTTP."""

from __future__ import annotations

import json
import sqlite3
import time
from functools import lru_cache
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from src.core.config import get_settings
from src.core.logging import get_logger, write_audit_log
from src.mcp_server.sql_guard import SQLGuardError, SQLTimeoutError, execute_safe_select
from src.retrieval.hybrid_search import HybridSearcher

logger = get_logger(__name__)
settings = get_settings()

mcp = FastMCP(
    "ai-papers-mcp",
    host=settings.mcp.host,
    port=settings.mcp.port,
    stateless_http=True,
)

VALID_PRIORITIES = {"low", "medium", "high"}


@lru_cache(maxsize=1)
def _get_searcher() -> HybridSearcher:
    return HybridSearcher(device="cpu")


def _open_ro(db_path: str | Path) -> sqlite3.Connection:
    resolved = Path(db_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")
    conn = sqlite3.connect(f"file:{resolved.as_posix()}?mode=ro", uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    return conn


# =============================================================================
# RESOURCES (3)
# =============================================================================


@mcp.resource("db://schema")
def resource_db_schema() -> str:
    """Return CREATE TABLE DDL of all 5 tables plus 2 sample rows per table."""
    cfg = get_settings()
    conn = _open_ro(cfg.env.papers_db_path)
    try:
        tables = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        sections: list[str] = []
        for tbl in tables:
            name = tbl["name"]
            ddl = tbl["sql"]
            sample_rows = [
                dict(r) for r in conn.execute(f'SELECT * FROM "{name}" LIMIT 2').fetchall()
            ]
            # Truncate long abstract values in sample preview to save prompt tokens
            for row in sample_rows:
                if "abstract" in row and isinstance(row["abstract"], str):
                    row["abstract"] = row["abstract"][:140] + "..."
            sections.append(
                f"-- Table: {name}\n{ddl};\n"
                f"-- Sample rows:\n{json.dumps(sample_rows, ensure_ascii=False, indent=2)}"
            )
        return "\n\n".join(sections)
    finally:
        conn.close()


@mcp.resource("papers://catalog")
def resource_papers_catalog() -> str:
    """Return compact catalog (arxiv_id, year, topic_cluster, title) of all indexed papers."""
    cfg = get_settings()
    conn = _open_ro(cfg.env.papers_db_path)
    try:
        rows = conn.execute(
            "SELECT arxiv_id, year, topic_cluster, title "
            "FROM papers ORDER BY topic_cluster, year DESC, arxiv_id"
        ).fetchall()
        lines = [
            f"- [{r['arxiv_id']}] ({r['year']}, {r['topic_cluster']}) {r['title']}"
            for r in rows
        ]
        return "\n".join(lines)
    finally:
        conn.close()


@mcp.resource("papers://stats")
def resource_papers_stats() -> str:
    """Return corpus statistics by year, topic cluster, and available benchmarks."""
    cfg = get_settings()
    conn = _open_ro(cfg.env.papers_db_path)
    try:
        total_papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        total_authors = conn.execute("SELECT COUNT(*) FROM authors").fetchone()[0]
        total_notes = conn.execute("SELECT COUNT(*) FROM reading_notes").fetchone()[0]
        by_year = dict(
            conn.execute(
                "SELECT year, COUNT(*) FROM papers GROUP BY year ORDER BY year"
            ).fetchall()
        )
        by_cluster = dict(
            conn.execute(
                "SELECT topic_cluster, COUNT(*) FROM papers "
                "GROUP BY topic_cluster ORDER BY topic_cluster"
            ).fetchall()
        )
        benchmarks = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT benchmark_name FROM benchmarks_reported ORDER BY benchmark_name"
            ).fetchall()
        ]
        return json.dumps(
            {
                "total_papers": total_papers,
                "total_authors": total_authors,
                "total_reading_notes": total_notes,
                "by_year": by_year,
                "by_topic_cluster": by_cluster,
                "benchmarks_available": benchmarks,
            },
            ensure_ascii=False,
            indent=2,
        )
    finally:
        conn.close()


# =============================================================================
# TOOLS (4)
# =============================================================================


@mcp.tool()
def query_papers_db(sql: str) -> dict[str, Any]:
    """Execute a read-only SQLite SELECT query on papers.db through the 4-layer SQL guard."""
    cfg = get_settings()
    t0 = time.perf_counter()
    try:
        rows, safe_sql = execute_safe_select(
            cfg.env.papers_db_path,
            sql,
            max_limit=cfg.mcp.sql_max_limit,
            timeout_sec=cfg.mcp.sql_timeout_seconds,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        write_audit_log(
            cfg.env.audit_log_path,
            tool="query_papers_db",
            args={"sql": sql},
            status="ok",
            duration_ms=duration_ms,
            row_count=len(rows),
        )
        return {
            "status": "ok",
            "executed_sql": safe_sql,
            "row_count": len(rows),
            "duration_ms": round(duration_ms, 2),
            "rows": rows,
        }
    except (SQLGuardError, SQLTimeoutError, FileNotFoundError) as exc:
        duration_ms = (time.perf_counter() - t0) * 1000.0
        write_audit_log(
            cfg.env.audit_log_path,
            tool="query_papers_db",
            args={"sql": sql},
            status="blocked",
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            "status": "error",
            "error": str(exc),
            "duration_ms": round(duration_ms, 2),
            "rows": [],
        }


@mcp.tool()
def search_paper_chunks(
    query: str,
    arxiv_ids: list[str] | None = None,
    year_min: int | None = None,
    only_tables: bool = False,
    top_k: int = 5,
) -> dict[str, Any]:
    """Search AI research paper chunks using Hybrid (Dense + BM25) + RRF + ONNX Reranker."""
    cfg = get_settings()
    t0 = time.perf_counter()
    safe_k = max(1, min(int(top_k), 15))
    try:
        searcher = _get_searcher()
        chunks = searcher.search(
            query,
            mode="hybrid_rerank",
            arxiv_ids=arxiv_ids,
            year_min=year_min,
            only_tables=only_tables,
            top_k=safe_k,
        )
        duration_ms = (time.perf_counter() - t0) * 1000.0
        write_audit_log(
            cfg.env.audit_log_path,
            tool="search_paper_chunks",
            args={
                "query": query,
                "arxiv_ids": arxiv_ids,
                "year_min": year_min,
                "only_tables": only_tables,
                "top_k": safe_k,
            },
            status="ok",
            duration_ms=duration_ms,
            row_count=len(chunks),
        )
        return {
            "status": "ok",
            "count": len(chunks),
            "duration_ms": round(duration_ms, 2),
            "chunks": chunks,
        }
    except Exception as exc:  # noqa: BLE001
        duration_ms = (time.perf_counter() - t0) * 1000.0
        write_audit_log(
            cfg.env.audit_log_path,
            tool="search_paper_chunks",
            args={"query": query},
            status="error",
            duration_ms=duration_ms,
            error=str(exc),
        )
        return {
            "status": "error",
            "error": str(exc),
            "duration_ms": round(duration_ms, 2),
            "chunks": [],
        }


@mcp.tool()
def get_paper_metadata(arxiv_id: str) -> dict[str, Any]:
    """Retrieve complete metadata, authors, benchmarks, and reading notes for one arXiv paper."""
    cfg = get_settings()
    t0 = time.perf_counter()
    conn = _open_ro(cfg.env.papers_db_path)
    try:
        paper = conn.execute(
            "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id.strip(),)
        ).fetchone()
        if paper is None:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            write_audit_log(
                cfg.env.audit_log_path,
                tool="get_paper_metadata",
                args={"arxiv_id": arxiv_id},
                status="not_found",
                duration_ms=duration_ms,
            )
            return {"status": "not_found", "error": f"Paper '{arxiv_id}' not found in papers.db."}

        authors = [
            r["full_name"]
            for r in conn.execute(
                "SELECT a.full_name FROM authors a "
                "JOIN paper_authors pa ON a.author_id = pa.author_id "
                "WHERE pa.arxiv_id = ? ORDER BY pa.author_order ASC",
                (arxiv_id.strip(),),
            ).fetchall()
        ]
        benchmarks = [
            dict(r)
            for r in conn.execute(
                "SELECT model_name, benchmark_name, metric_name, score, notes "
                "FROM benchmarks_reported WHERE arxiv_id = ? ORDER BY benchmark_name, score DESC",
                (arxiv_id.strip(),),
            ).fetchall()
        ]
        notes = [
            dict(r)
            for r in conn.execute(
                "SELECT note_id, note_text, tags_json, priority, created_at "
                "FROM reading_notes WHERE arxiv_id = ? ORDER BY created_at DESC",
                (arxiv_id.strip(),),
            ).fetchall()
        ]
        duration_ms = (time.perf_counter() - t0) * 1000.0
        write_audit_log(
            cfg.env.audit_log_path,
            tool="get_paper_metadata",
            args={"arxiv_id": arxiv_id},
            status="ok",
            duration_ms=duration_ms,
            row_count=1,
        )
        return {
            "status": "ok",
            "duration_ms": round(duration_ms, 2),
            "paper": dict(paper),
            "authors": authors,
            "benchmarks": benchmarks,
            "reading_notes": notes,
        }
    finally:
        conn.close()


@mcp.tool()
def create_reading_note(
    arxiv_id: str,
    note: str,
    tags: list[str] | None = None,
    priority: str = "medium",
) -> dict[str, Any]:
    """Insert a reading note into reading_notes (intended to run only after HITL approval)."""
    cfg = get_settings()
    t0 = time.perf_counter()
    clean_id = arxiv_id.strip()
    clean_priority = priority.strip().lower()
    clean_tags = [str(t).strip() for t in (tags or []) if str(t).strip()]

    if clean_priority not in VALID_PRIORITIES:
        err = f"Invalid priority '{priority}'. Must be one of {sorted(VALID_PRIORITIES)}."
        write_audit_log(
            cfg.env.audit_log_path,
            tool="create_reading_note",
            args={"arxiv_id": clean_id, "priority": priority},
            status="blocked",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            error=err,
        )
        return {"status": "error", "error": err}

    if not note or not note.strip():
        err = "Note text must not be empty."
        write_audit_log(
            cfg.env.audit_log_path,
            tool="create_reading_note",
            args={"arxiv_id": clean_id},
            status="blocked",
            duration_ms=(time.perf_counter() - t0) * 1000.0,
            error=err,
        )
        return {"status": "error", "error": err}

    conn = sqlite3.connect(str(Path(cfg.env.papers_db_path).resolve()), timeout=5.0)
    try:
        conn.execute("PRAGMA foreign_keys = ON;")
        exists = conn.execute(
            "SELECT title FROM papers WHERE arxiv_id = ?", (clean_id,)
        ).fetchone()
        if exists is None:
            err = f"Cannot create reading note: paper '{clean_id}' does not exist in papers.db."
            write_audit_log(
                cfg.env.audit_log_path,
                tool="create_reading_note",
                args={"arxiv_id": clean_id, "priority": clean_priority},
                status="blocked",
                duration_ms=(time.perf_counter() - t0) * 1000.0,
                error=err,
            )
            return {"status": "error", "error": err}

        cur = conn.execute(
            "INSERT INTO reading_notes (arxiv_id, note_text, tags_json, priority) "
            "VALUES (?, ?, ?, ?)",
            (clean_id, note.strip(), json.dumps(clean_tags, ensure_ascii=False), clean_priority),
        )
        conn.commit()
        note_id = cur.lastrowid
        duration_ms = (time.perf_counter() - t0) * 1000.0
        write_audit_log(
            cfg.env.audit_log_path,
            tool="create_reading_note",
            args={"arxiv_id": clean_id, "tags": clean_tags, "priority": clean_priority},
            status="ok",
            duration_ms=duration_ms,
            row_count=1,
        )
        return {
            "status": "ok",
            "note_id": note_id,
            "arxiv_id": clean_id,
            "paper_title": exists[0],
            "priority": clean_priority,
            "tags": clean_tags,
            "duration_ms": round(duration_ms, 2),
        }
    finally:
        conn.close()


# =============================================================================
# PROMPTS (1)
# =============================================================================


@mcp.prompt()
def compare_papers_methodology(arxiv_id_a: str, arxiv_id_b: str) -> str:
    """Standardized prompt guiding structured comparison of two indexed AI research papers."""
    return (
        f"Compare the AI research papers arXiv:{arxiv_id_a} and arXiv:{arxiv_id_b} using verified "
        f"evidence from both `get_paper_metadata` and `search_paper_chunks`.\n"
        f"Structure your analysis into:\n"
        f"1. Core Architecture & Key Mechanism\n"
        f"2. Training / Inference Efficiency & Resource Trade-offs\n"
        f"3. Datasets & Quantitative Benchmark Comparison (include table numbers if available)\n"
        f"4. Key Limitations & When to Prefer Each Method\n"
        f"Attach inline citations [arXiv:XXXX.XXXXX, p. N] or [SQL: table_name] after every claim."
    )


def main() -> None:
    cfg = get_settings()
    transport = cfg.effective_mcp_transport
    logger.info(
        f"Starting FastMCP server (transport={transport}, host={cfg.mcp.host}, port={cfg.mcp.port})"
    )
    mcp.run(transport=transport)


if __name__ == "__main__":
    main()
