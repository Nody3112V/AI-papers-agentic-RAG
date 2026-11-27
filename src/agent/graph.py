"""LangGraph 8-node StateGraph compiler backed by AsyncSqliteSaver (checkpoints.db + WAL)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from src.agent.mcp_client import MCPClient
from src.agent.nodes import INSUFFICIENT_EVIDENCE_MSG, make_nodes
from src.agent.state import AgentState
from src.core.config import get_settings


def _route_after_router(state: AgentState) -> str:
    decision = state.get("route_decision") or {}
    route = decision.get("route", "rag_only")
    if route == "direct":
        return "citation_generator"
    if route == "write_note":
        return "hitl_write_approval"
    if route == "mcp_sql":
        return "mcp_sql_exec"
    if route == "hybrid":
        if decision.get("hybrid_mode") == "parallel":
            return "hybrid_parallel_exec"
        return "mcp_sql_exec"  # sql_then_rag runs SQL first, then RAG
    return "rag_retrieve"


def _route_after_sql(state: AgentState) -> str:
    decision = state.get("route_decision") or {}
    if decision.get("route") == "hybrid":
        return "rag_retrieve"
    return "citation_generator"


def _route_after_grader(state: AgentState) -> str:
    cfg = get_settings()
    chunks = state.get("retrieved_chunks") or []
    retries = int(state.get("retry_count", 0))
    if len(chunks) == 0 and retries < cfg.agent.max_rag_retries:
        return "query_rewriter"
    return "citation_generator"


def _route_after_generator(state: AgentState) -> str:
    decision = state.get("route_decision") or {}
    if decision.get("route") == "direct":
        return END
    if state.get("draft_answer") == INSUFFICIENT_EVIDENCE_MSG:
        return END
    return "hallucination_checker"


def _route_after_hallucination(state: AgentState) -> str:
    if state.get("is_grounded", True):
        return END
    if state.get("draft_answer") == INSUFFICIENT_EVIDENCE_MSG:
        return END
    return "citation_generator"


def build_agent_graph(mcp_client: MCPClient | None = None) -> StateGraph:
    client = mcp_client or MCPClient()
    nodes = make_nodes(client)

    builder = StateGraph(AgentState)
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    builder.add_edge(START, "contextualize_query")
    builder.add_edge("contextualize_query", "router")
    builder.add_conditional_edges(
        "router",
        _route_after_router,
        {
            "citation_generator": "citation_generator",
            "hitl_write_approval": "hitl_write_approval",
            "mcp_sql_exec": "mcp_sql_exec",
            "hybrid_parallel_exec": "hybrid_parallel_exec",
            "rag_retrieve": "rag_retrieve",
        },
    )
    builder.add_conditional_edges(
        "mcp_sql_exec",
        _route_after_sql,
        {
            "rag_retrieve": "rag_retrieve",
            "citation_generator": "citation_generator",
        },
    )
    builder.add_edge("hybrid_parallel_exec", "batch_document_grader")
    builder.add_edge("rag_retrieve", "batch_document_grader")
    builder.add_conditional_edges(
        "batch_document_grader",
        _route_after_grader,
        {
            "query_rewriter": "query_rewriter",
            "citation_generator": "citation_generator",
        },
    )
    builder.add_edge("query_rewriter", "rag_retrieve")
    builder.add_edge("hitl_write_approval", END)
    builder.add_conditional_edges(
        "citation_generator",
        _route_after_generator,
        {
            "hallucination_checker": "hallucination_checker",
            END: END,
        },
    )
    builder.add_conditional_edges(
        "hallucination_checker",
        _route_after_hallucination,
        {
            "citation_generator": "citation_generator",
            END: END,
        },
    )
    return builder


class AgentGraphRuntime:
    """Async lifecycle manager for AsyncSqliteSaver (WAL mode) + compiled LangGraph."""

    def __init__(
        self,
        *,
        checkpoints_db_path: str | Path | None = None,
        mcp_client: MCPClient | None = None,
    ) -> None:
        cfg = get_settings()
        self.checkpoints_db_path = Path(checkpoints_db_path or cfg.env.checkpoints_db_path)
        self.mcp_client = mcp_client or MCPClient()
        self._conn: aiosqlite.Connection | None = None
        self.checkpointer: AsyncSqliteSaver | None = None
        self.graph: Any = None

    async def start(self) -> Any:
        self.checkpoints_db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.checkpoints_db_path))
        await self._conn.execute("PRAGMA journal_mode = WAL;")
        await self._conn.commit()
        self.checkpointer = AsyncSqliteSaver(self._conn)
        await self.checkpointer.setup()
        builder = build_agent_graph(mcp_client=self.mcp_client)
        self.graph = builder.compile(checkpointer=self.checkpointer)
        return self.graph

    async def stop(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> Any:
        return await self.start()

    async def __aexit__(self, *args: object) -> None:
        await self.stop()
