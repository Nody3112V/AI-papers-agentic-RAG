"""Initialize the 5-table SQLite schema for data/sqlite/papers.db with WAL mode."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_settings

SCHEMA_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS papers (
    arxiv_id          TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    abstract          TEXT NOT NULL,
    published_date    TEXT NOT NULL,
    year              INTEGER NOT NULL,
    primary_category  TEXT NOT NULL,
    topic_cluster     TEXT NOT NULL,
    venue             TEXT DEFAULT 'arXiv',
    page_count        INTEGER,
    pdf_path          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS authors (
    author_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name         TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS paper_authors (
    arxiv_id          TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    author_id         INTEGER NOT NULL REFERENCES authors(author_id) ON DELETE CASCADE,
    author_order      INTEGER NOT NULL,
    PRIMARY KEY (arxiv_id, author_id)
);

CREATE TABLE IF NOT EXISTS benchmarks_reported (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id          TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    model_name        TEXT NOT NULL,
    benchmark_name    TEXT NOT NULL,
    metric_name       TEXT NOT NULL,
    score             REAL NOT NULL,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS reading_notes (
    note_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    arxiv_id          TEXT NOT NULL REFERENCES papers(arxiv_id) ON DELETE CASCADE,
    note_text         TEXT NOT NULL,
    tags_json         TEXT NOT NULL DEFAULT '[]',
    priority          TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high')),
    created_at        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_papers_year_topic ON papers(year, topic_cluster);
CREATE INDEX IF NOT EXISTS idx_bench_name_score ON benchmarks_reported(benchmark_name, score DESC);
CREATE INDEX IF NOT EXISTS idx_notes_arxiv ON reading_notes(arxiv_id);
"""


def init_papers_db(db_path: str | Path) -> Path:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(SCHEMA_DDL)
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name;"
            ).fetchall()
        ]
        print(f"[DB READY] path={path} | journal_mode={mode} | tables={tables}")
    finally:
        conn.close()
    return path


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Initialize papers.db SQLite schema.")
    parser.add_argument(
        "--db-path",
        default=settings.env.papers_db_path,
        help="Path to SQLite papers.db file.",
    )
    args = parser.parse_args()
    init_papers_db(args.db_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
