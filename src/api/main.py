"""FastAPI SSE Gateway exposing /chat/stream, /chat/resume, /sessions/{thread_id}, and /health."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sse_starlette.sse import EventSourceResponse

from src.agent.graph import AgentGraphRuntime
from src.agent.mcp_client import MCPClient
from src.api.guardrails import InputGuardrailError, limiter, validate_user_query
from src.api.schemas import (
    ChatResumeRequest,
    ChatStreamRequest,
    HealthResponse,
    SessionMessage,
    SessionResponse,
)
from src.core.config import get_settings
from src.core.logging import get_logger
from src.ingestion.indexer import get_qdrant_client
from src.llm.provider import get_ordered_models

logger = get_logger(__name__)

NODE_STATUS_LABELS: dict[str, str] = {
    "contextualize_query": "Contextualizing conversation query...",
    "router": "Routing query (SQL / RAG / Hybrid / Write / Direct)...",
    "rag_retrieve": "Searching AI papers (Hybrid Dense+BM25 + RRF + ONNX Reranker)...",
    "mcp_sql_exec": "Querying papers.db through 4-layer SQL guard...",
    "hybrid_parallel_exec": "Running SQL + Hybrid RAG in parallel...",
    "batch_document_grader": "Grading retrieved paper chunks in a single batch...",
    "query_rewriter": "Rewriting query with formal academic ML/NLP terminology...",
    "hitl_write_approval": "Waiting for Human-in-the-Loop write approval...",
    "citation_generator": "Synthesizing grounded answer with inline citations...",
    "hallucination_checker": "Verifying citations and factual grounding (2-tier checker)...",
}


def _setup_tracing_if_enabled() -> None:
    cfg = get_settings()
    if not cfg.env.enable_tracing:
        return
    try:
        from openinference.instrumentation.langchain import LangChainInstrumentor
        from phoenix.otel import register

        headers = (
            {"api_key": cfg.env.phoenix_api_key}
            if cfg.env.phoenix_api_key
            else None
        )
        tracer_provider = register(
            project_name=cfg.app.name,
            endpoint=cfg.env.phoenix_collector_endpoint,
            headers=headers,
            batch=True,
        )
        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info(
            f"Arize Phoenix OTLP tracing enabled -> {cfg.env.phoenix_collector_endpoint}"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Failed to initialize Phoenix tracing ({exc}); continuing without tracing.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_tracing_if_enabled()
    mcp_client = getattr(app.state, "mcp_client_override", None) or MCPClient()
    checkpoints_path = getattr(app.state, "checkpoints_db_override", None)
    runtime = AgentGraphRuntime(
        checkpoints_db_path=checkpoints_path,
        mcp_client=mcp_client,
    )
    app.state.runtime = runtime
    app.state.graph = await runtime.start()
    logger.info("FastAPI Agentic RAG Gateway started.")
    try:
        yield
    finally:
        await runtime.stop()
        logger.info("FastAPI Agentic RAG Gateway stopped.")


app = FastAPI(
    title="AI Papers Agentic RAG + MCP Gateway",
    version="3.0.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: str, data: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(data, ensure_ascii=False)}


def _chunk_text_for_stream(text: str, words_per_chunk: int = 8) -> list[str]:
    if not text:
        return []
    tokens = text.split(" ")
    chunks: list[str] = []
    for i in range(0, len(tokens), words_per_chunk):
        piece = " ".join(tokens[i : i + words_per_chunk])
        if i + words_per_chunk < len(tokens):
            piece += " "
        chunks.append(piece)
    return chunks


async def _stream_graph_execution(
    graph: Any,
    graph_input: Any,
    *,
    thread_id: str,
) -> AsyncIterator[dict[str, str]]:
    t_start = time.perf_counter()
    ttft_ms: float | None = None
    emitted_tools = 0
    final_answer = ""
    final_citations: list[dict[str, Any]] = []
    is_grounded = True
    paused_for_hitl = False

    config = {"configurable": {"thread_id": thread_id}}

    yield _sse("status", {"node": "start", "message": "Processing request..."})

    async for update in graph.astream(graph_input, config=config, stream_mode="updates"):
        if "__interrupt__" in update:
            paused_for_hitl = True
            interrupts = update["__interrupt__"]
            first = interrupts[0] if interrupts else None
            payload_obj = getattr(first, "value", first) if first is not None else {}
            if not isinstance(payload_obj, dict):
                payload_obj = {"action": "create_reading_note", "payload": {}}
            yield _sse(
                "hitl_required",
                {
                    "thread_id": thread_id,
                    "action": payload_obj.get("action", "create_reading_note"),
                    "payload": payload_obj.get("payload", {}),
                },
            )
            continue

        for node_name, node_state in update.items():
            if not isinstance(node_state, dict):
                continue

            label = NODE_STATUS_LABELS.get(node_name, f"Running {node_name}...")
            if node_name == "router" and isinstance(node_state.get("route_decision"), dict):
                rd = node_state["route_decision"]
                hmode = f" / {rd['hybrid_mode']}" if rd.get("hybrid_mode") else ""
                label = f"Routed query -> {rd.get('route')}{hmode}"
            yield _sse("status", {"node": node_name, "message": label})

            # Emit any newly appended MCP tool calls
            tool_logs = node_state.get("tool_calls_log") or []
            if len(tool_logs) > emitted_tools:
                for entry in tool_logs[emitted_tools:]:
                    yield _sse("tool_call", entry)
                emitted_tools = len(tool_logs)

            if node_state.get("citations"):
                final_citations = node_state["citations"]
            if node_state.get("draft_answer"):
                final_answer = str(node_state["draft_answer"])
            if "is_grounded" in node_state:
                is_grounded = bool(node_state["is_grounded"])

    if paused_for_hitl:
        elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
        yield _sse(
            "done",
            {
                "latency_ms": elapsed_ms,
                "ttft_ms": ttft_ms,
                "is_grounded": True,
                "paused_for_hitl": True,
            },
        )
        return

    # Stream answer tokens smoothly
    if final_answer:
        for delta in _chunk_text_for_stream(final_answer):
            if ttft_ms is None:
                ttft_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
            yield _sse("token", {"delta": delta})

    if final_citations:
        yield _sse("citations", {"items": final_citations})

    elapsed_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
    yield _sse(
        "done",
        {
            "latency_ms": elapsed_ms,
            "ttft_ms": ttft_ms if ttft_ms is not None else elapsed_ms,
            "is_grounded": is_grounded,
            "paused_for_hitl": False,
        },
    )


@app.post("/api/v1/chat/stream")
@limiter.limit("20/minute")
async def chat_stream(request: Request, body: ChatStreamRequest) -> EventSourceResponse:
    cfg = get_settings()
    try:
        clean_query = validate_user_query(body.query, max_chars=cfg.app.max_query_chars)
    except InputGuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    graph_input: dict[str, Any] = {
        "messages": [HumanMessage(content=clean_query)],
        "thread_id": body.thread_id,
        "original_query": clean_query,
    }
    return EventSourceResponse(
        _stream_graph_execution(app.state.graph, graph_input, thread_id=body.thread_id)
    )


@app.post("/api/v1/chat/resume")
@limiter.limit("20/minute")
async def chat_resume(request: Request, body: ChatResumeRequest) -> EventSourceResponse:
    resume_payload: dict[str, Any] = {"decision": body.decision}
    if body.edited_payload is not None:
        resume_payload["edited_payload"] = body.edited_payload
    if body.reject_reason is not None:
        resume_payload["reject_reason"] = body.reject_reason

    return EventSourceResponse(
        _stream_graph_execution(
            app.state.graph,
            Command(resume=resume_payload),
            thread_id=body.thread_id,
        )
    )


@app.get("/api/v1/sessions/{thread_id}", response_model=SessionResponse)
async def get_session(thread_id: str) -> SessionResponse:
    snapshot = await app.state.graph.aget_state({"configurable": {"thread_id": thread_id}})
    values = snapshot.values if snapshot and snapshot.values else {}
    raw_messages = values.get("messages") or []
    messages: list[SessionMessage] = []
    for msg in raw_messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant" if isinstance(msg, AIMessage) else "system"
        messages.append(SessionMessage(role=role, content=str(getattr(msg, "content", msg))))

    pending_hitl: dict[str, Any] | None = None
    top_interrupts = getattr(snapshot, "interrupts", None) or []
    if top_interrupts:
        val = getattr(top_interrupts[0], "value", top_interrupts[0])
        if isinstance(val, dict):
            pending_hitl = val

    if pending_hitl is None:
        tasks = getattr(snapshot, "tasks", None) or []
        for task in tasks:
            interrupts = getattr(task, "interrupts", None) or []
            if interrupts:
                val = getattr(interrupts[0], "value", interrupts[0])
                if isinstance(val, dict):
                    pending_hitl = val
                    break

    if (
        pending_hitl is None
        and "hitl_write_approval" in (getattr(snapshot, "next", None) or ())
        and isinstance(values.get("pending_action"), dict)
    ):
        pending_hitl = {
            "action": "create_reading_note",
            "payload": values["pending_action"],
        }

    return SessionResponse(
        thread_id=thread_id,
        pending_hitl=pending_hitl,
        messages=messages,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> JSONResponse:
    cfg = get_settings()
    services: dict[str, dict[str, Any]] = {}
    all_ok = True

    # 1. Qdrant check
    try:
        qclient = get_qdrant_client(mode=cfg.effective_qdrant_mode)
        cols = [c.name for c in qclient.get_collections().collections]
        services["qdrant"] = {"ok": True, "mode": cfg.effective_qdrant_mode, "collections": cols}
    except Exception as exc:  # noqa: BLE001
        all_ok = False
        services["qdrant"] = {"ok": False, "error": str(exc)}

    # 2. MCP Server check
    if cfg.env.mcp_server_url.lower() == "inprocess":
        services["mcp"] = {"ok": True, "mode": "inprocess"}
    else:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(cfg.env.mcp_server_url)
                # Streamable HTTP MCP endpoint responds (e.g., 405/406/200) when server is reachable
                services["mcp"] = {"ok": resp.status_code < 500, "http_status": resp.status_code}
                if resp.status_code >= 500:
                    all_ok = False
        except Exception as exc:  # noqa: BLE001
            all_ok = False
            services["mcp"] = {"ok": False, "error": str(exc)}

    # 3. LLM Provider check
    try:
        providers = [name for name, _ in get_ordered_models(tier="fast")]
        services["llm"] = {"ok": bool(providers), "providers": providers}
    except Exception as exc:  # noqa: BLE001
        all_ok = False
        services["llm"] = {"ok": False, "error": str(exc)}

    rss_mb = round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    payload = HealthResponse(
        status="healthy" if all_ok else "degraded",
        services=services,
        memory_rss_mb=rss_mb,
    )
    return JSONResponse(status_code=200 if all_ok else 503, content=payload.model_dump())
