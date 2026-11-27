"""4-Layer SQLite Read-Only Security Guard (sqlglot AST, LIMIT cap, ro URI, progress_handler)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

FORBIDDEN_TABLES = {
    "sqlite_master",
    "sqlite_schema",
    "sqlite_temp_master",
    "sqlite_temp_schema",
}


class SQLGuardError(ValueError):
    """Raised when a SQL query violates the 4-layer read-only security guard."""


class SQLTimeoutError(TimeoutError):
    """Raised when a SQLite SELECT query exceeds the execution timeout."""


def validate_and_rewrite_sql(sql: str, *, max_limit: int = 50) -> str:
    """Layer 1 & 2: Validate single SELECT AST, block recursion/system tables, cap LIMIT."""
    if not sql or not sql.strip():
        raise SQLGuardError("SQL query must not be empty.")

    try:
        statements = [stmt for stmt in sqlglot.parse(sql, read="sqlite") if stmt is not None]
    except ParseError as exc:
        raise SQLGuardError(f"Invalid SQL syntax: {exc}") from exc

    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise SQLGuardError("Only a single read-only SELECT statement is allowed.")

    tree = statements[0]

    # Block any nested write/DDL expressions just in case
    for forbidden_cls in (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Drop,
        exp.Alter,
        exp.Create,
        exp.Command,
    ):
        if tree.find(forbidden_cls) is not None:
            raise SQLGuardError(f"Forbidden SQL operation: {forbidden_cls.__name__}")

    # Block WITH RECURSIVE (prevents infinite CPU loops)
    for cte_with in tree.find_all(exp.With):
        if cte_with.args.get("recursive"):
            raise SQLGuardError("WITH RECURSIVE is forbidden to prevent infinite loops.")

    # Block SQLite internal system tables & pragma table-valued functions
    for tbl in tree.find_all(exp.Table):
        tbl_name = tbl.name.lower()
        if tbl_name in FORBIDDEN_TABLES or tbl_name.startswith("pragma_"):
            raise SQLGuardError(f"Access to system table '{tbl.name}' is forbidden.")

    # Layer 2: Enforce LIMIT <= max_limit
    limit_expr = tree.args.get("limit")
    if limit_expr is None:
        tree = tree.limit(max_limit)
    else:
        raw_val = limit_expr.expression
        try:
            limit_val = int(raw_val.name)
            if limit_val <= 0 or limit_val > max_limit:
                tree = tree.limit(max_limit)
        except (AttributeError, TypeError, ValueError):
            tree = tree.limit(max_limit)

    return tree.sql(dialect="sqlite")


def execute_safe_select(
    db_path: str | Path,
    sql: str,
    *,
    max_limit: int = 50,
    timeout_sec: float = 2.0,
) -> tuple[list[dict[str, Any]], str]:
    """Execute a validated read-only SELECT with URI ?mode=ro, query_only, and real timeout."""
    safe_sql = validate_and_rewrite_sql(sql, max_limit=max_limit)
    resolved = Path(db_path).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"SQLite database not found: {resolved}")

    uri = f"file:{resolved.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    start = time.monotonic()

    def _progress_callback() -> int:
        return 1 if (time.monotonic() - start) > timeout_sec else 0

    conn.set_progress_handler(_progress_callback, 10000)
    try:
        conn.execute("PRAGMA query_only = ON;")
        rows = conn.execute(safe_sql).fetchall()
        return [dict(row) for row in rows], safe_sql
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise SQLTimeoutError(
                f"SQL query aborted after exceeding {timeout_sec:.1f}s timeout."
            ) from exc
        raise SQLGuardError(f"SQL execution error: {exc}") from exc
    finally:
        conn.set_progress_handler(None, 0)
        conn.close()
