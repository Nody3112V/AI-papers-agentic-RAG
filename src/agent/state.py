"""LangGraph AgentState contract and Pydantic Structured Output schemas."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["rag_only", "mcp_sql", "hybrid", "write_note", "direct"]
    hybrid_mode: Literal["parallel", "sql_then_rag"] | None = None
    sql_query: str | None = Field(
        default=None,
        description="Single valid read-only SQLite SELECT query with LIMIT <= 50 if SQL is needed.",
    )
    arxiv_ids: list[str] | None = None
    year_min: int | None = None
    only_tables: bool = False
    write_payload: dict | None = Field(
        default=None,
        description="For write_note: {'arxiv_id': str, 'note': str, 'tags': list[str], 'priority': str}",
    )
    reasoning: str


class ChunkGrade(BaseModel):
    chunk_index: int
    relevant: bool
    reason: str


class BatchGradeOutput(BaseModel):
    grades: list[ChunkGrade]


class HallucinationResult(BaseModel):
    is_grounded: bool
    unsupported_claims: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    thread_id: str
    original_query: str
    standalone_query: str
    rewritten_query: str | None
    route_decision: dict | None
    retrieved_chunks: list[dict]
    sql_results: list[dict] | None
    tool_calls_log: list[dict]
    pending_action: dict | None
    retry_count: int
    sql_retry_count: int
    gen_retry_count: int
    draft_answer: str
    citations: list[dict]
    is_grounded: bool
    error_trace: list[str]
