"""FastAPI request/response models and SSE event payload contracts."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatStreamRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1)
    mode_override: Literal["rag_only", "mcp_sql", "hybrid", "write_note", "direct"] | None = None


class ChatResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1, max_length=128)
    decision: Literal["approve", "reject", "edit"]
    edited_payload: dict[str, Any] | None = None
    reject_reason: str | None = None


class SessionMessage(BaseModel):
    role: str
    content: str


class SessionResponse(BaseModel):
    thread_id: str
    pending_hitl: dict[str, Any] | None = None
    messages: list[SessionMessage] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    services: dict[str, dict[str, Any]]
    memory_rss_mb: float
