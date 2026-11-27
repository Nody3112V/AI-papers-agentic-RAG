"""Deterministic integration tests for LangGraph HITL 3-branch lifecycle (approve / edit / reject)."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from scripts.init_db import init_papers_db
from src.agent import nodes as agent_nodes
from src.agent.graph import AgentGraphRuntime
from src.agent.mcp_client import MCPClient, clear_resource_cache
from src.agent.state import RouteDecision
from src.core.config import get_settings


@pytest.fixture()
def temp_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    papers_db = tmp_path / "papers.db"
    checkpoints_db = tmp_path / "checkpoints.db"
    audit_log = tmp_path / "mcp_audit.jsonl"
    init_papers_db(papers_db)

    conn = sqlite3.connect(str(papers_db))
    conn.execute(
        "INSERT INTO papers (arxiv_id, title, abstract, published_date, year, "
        "primary_category, topic_cluster, venue, page_count, pdf_path) "
        "VALUES ('2310.11511', 'Self-RAG', 'Self-reflective RAG.', '2023-10-17', 2023, "
        "'cs.CL', 'rag', 'ICLR 2024', 30, '2310.11511.pdf')"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("PAPERS_DB_PATH", str(papers_db))
    monkeypatch.setenv("CHECKPOINTS_DB_PATH", str(checkpoints_db))
    monkeypatch.setenv("AUDIT_LOG_PATH", str(audit_log))
    get_settings.cache_clear()
    clear_resource_cache()

    async def _fake_structured(schema: Any, messages: Any, *, tier: str = "fast") -> Any:
        if schema is RouteDecision:
            return RouteDecision(
                route="write_note",
                write_payload={
                    "arxiv_id": "2310.11511",
                    "note": "Initial draft note.",
                    "tags": ["rag"],
                    "priority": "high",
                },
                reasoning="User requested saving a reading note.",
            )
        raise AssertionError(f"Unexpected schema call: {schema}")

    monkeypatch.setattr(agent_nodes, "ainvoke_structured", _fake_structured)
    yield papers_db, checkpoints_db
    get_settings.cache_clear()
    clear_resource_cache()


async def test_hitl_three_branches_approve_edit_reject(temp_env: tuple[Path, Path]) -> None:
    papers_db, checkpoints_db = temp_env
    mcp_client = MCPClient(inprocess=True)

    async with AgentGraphRuntime(
        checkpoints_db_path=checkpoints_db,
        mcp_client=mcp_client,
    ) as graph:
        # 1. APPROVE branch -> DB gets 1 row with priority='high'
        cfg_a = {"configurable": {"thread_id": "hitl_approve"}}
        paused_a = await graph.ainvoke(
            {"messages": [HumanMessage(content="Save note")], "thread_id": "hitl_approve", "original_query": "Save note"},
            config=cfg_a,
        )
        assert "__interrupt__" in paused_a
        res_a = await graph.ainvoke(Command(resume={"decision": "approve"}), config=cfg_a)
        assert "Saved reading note #1" in res_a["draft_answer"]

        # 2. EDIT branch -> DB gets row #2 with edited priority='medium' and edited note
        cfg_e = {"configurable": {"thread_id": "hitl_edit"}}
        paused_e = await graph.ainvoke(
            {"messages": [HumanMessage(content="Save note")], "thread_id": "hitl_edit", "original_query": "Save note"},
            config=cfg_e,
        )
        assert "__interrupt__" in paused_e
        res_e = await graph.ainvoke(
            Command(
                resume={
                    "decision": "edit",
                    "edited_payload": {
                        "priority": "medium",
                        "note": "Edited note for Project 2.",
                        "tags": ["rag", "adaptive"],
                    },
                }
            ),
            config=cfg_e,
        )
        assert "Saved reading note #2" in res_e["draft_answer"]
        assert "medium" in res_e["draft_answer"]

        # 3. REJECT branch -> DB remains at 2 rows (no write)
        cfg_r = {"configurable": {"thread_id": "hitl_reject"}}
        paused_r = await graph.ainvoke(
            {"messages": [HumanMessage(content="Save note")], "thread_id": "hitl_reject", "original_query": "Save note"},
            config=cfg_r,
        )
        assert "__interrupt__" in paused_r
        res_r = await graph.ainvoke(
            Command(resume={"decision": "reject", "reject_reason": "Already saved."}),
            config=cfg_r,
        )
        assert "cancelled" in res_r["draft_answer"].lower()

    conn = sqlite3.connect(str(papers_db))
    rows = conn.execute("SELECT note_text, priority FROM reading_notes ORDER BY note_id").fetchall()
    conn.close()
    assert rows == [
        ("Initial draft note.", "high"),
        ("Edited note for Project 2.", "medium"),
    ]
