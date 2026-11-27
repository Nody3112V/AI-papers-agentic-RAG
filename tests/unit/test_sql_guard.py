"""Unit tests for the 4-layer SQL security guard (all 10+ attack vectors + timeout)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.mcp_server.sql_guard import (
    SQLGuardError,
    SQLTimeoutError,
    execute_safe_select,
    validate_and_rewrite_sql,
)


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    db = tmp_path / "test_papers.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE papers (arxiv_id TEXT PRIMARY KEY, year INTEGER, title TEXT)")
    conn.executemany(
        "INSERT INTO papers VALUES (?, ?, ?)",
        [(f"2401.{i:05d}", 2024, f"Paper {i}") for i in range(120)],
    )
    conn.commit()
    conn.close()
    return db


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "DROP TABLE papers",
        "UPDATE papers SET year = 2000",
        "DELETE FROM papers WHERE year = 2024",
        "INSERT INTO papers VALUES ('x', 2025, 'hack')",
        "ATTACH DATABASE '/tmp/evil.db' AS evil",
        "PRAGMA table_info(papers)",
        "SELECT * FROM papers; DROP TABLE papers",
        "WITH RECURSIVE cnt(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM cnt) SELECT x FROM cnt",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM sqlite_schema",
    ],
)
def test_sql_guard_blocks_all_attack_vectors(malicious_sql: str) -> None:
    with pytest.raises(SQLGuardError):
        validate_and_rewrite_sql(malicious_sql)


def test_sql_guard_enforces_limit_cap(sample_db: Path) -> None:
    rows, safe_sql = execute_safe_select(sample_db, "SELECT * FROM papers LIMIT 9999", max_limit=50)
    assert len(rows) == 50
    assert "LIMIT 50" in safe_sql.upper()

    rows_no_limit, safe_sql_2 = execute_safe_select(sample_db, "SELECT arxiv_id FROM papers", max_limit=50)
    assert len(rows_no_limit) == 50
    assert "LIMIT 50" in safe_sql_2.upper()


def test_sql_guard_preserves_smaller_limit(sample_db: Path) -> None:
    rows, safe_sql = execute_safe_select(sample_db, "SELECT arxiv_id FROM papers LIMIT 5", max_limit=50)
    assert len(rows) == 5
    assert "LIMIT 5" in safe_sql.upper()


def test_sql_guard_aborts_expensive_cross_join_timeout(sample_db: Path) -> None:
    # 120^4 = 207 million combinations -> guaranteed to trigger progress_handler timeout
    heavy_sql = (
        "SELECT COUNT(*) AS c FROM papers a "
        "CROSS JOIN papers b CROSS JOIN papers c CROSS JOIN papers d"
    )
    with pytest.raises(SQLTimeoutError):
        execute_safe_select(sample_db, heavy_sql, timeout_sec=0.15)
