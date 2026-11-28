"""Chainlit UI for AI Papers Agentic RAG + MCP: Reasoning Steps, Side Citations, and 3-Button HITL."""

from __future__ import annotations

import json
import os
import uuid
from typing import Any

import chainlit as cl
import httpx

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


async def _iter_sse(response: httpx.Response):
    """Parse standard Server-Sent Events (event: + data:) from an httpx streaming response."""
    event_name = "message"
    data_lines: list[str] = []
    async for raw_line in response.aiter_lines():
        line = raw_line.strip("\r\n")
        if not line:
            if data_lines:
                raw_data = "\n".join(data_lines)
                try:
                    yield event_name, json.loads(raw_data)
                except json.JSONDecodeError:
                    yield event_name, {"raw": raw_data}
                event_name = "message"
                data_lines = []
            continue
        if line.startswith("event:"):
            event_name = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            data_lines.append(line.split(":", 1)[1].strip())


async def _consume_stream(endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Stream SSE events into Chainlit UI; returns hitl_required payload if paused."""
    hitl_event: dict[str, Any] | None = None
    citations_items: list[dict[str, Any]] = []
    done_meta: dict[str, Any] = {}
    answer_msg: cl.Message | None = None

    async with (
        httpx.AsyncClient(timeout=120.0) as client,
        client.stream("POST", f"{API_BASE_URL}{endpoint}", json=payload) as resp,
    ):
            if resp.status_code != 200:
                body = await resp.aread()
                await cl.Message(
                    content=f"**API Error ({resp.status_code}):** {body.decode('utf-8', errors='replace')}"
                ).send()
                return None

            async for event, data in _iter_sse(resp):
                if event == "status":
                    node = data.get("node", "agent")
                    if node == "start":
                        continue
                    async with cl.Step(name=f"Node: {node}", type="run") as step:
                        step.output = data.get("message", "")

                elif event == "tool_call":
                    tool_name = data.get("tool", "mcp_tool")
                    dur = data.get("duration_ms", 0)
                    status = data.get("status", "ok")
                    args = data.get("args", {})
                    async with cl.Step(name=f"MCP Tool: {tool_name} ({dur} ms)", type="tool") as step:
                        if tool_name == "query_papers_db" and "sql" in args:
                            step.input = f"```sql\n{args['sql']}\n```"
                        else:
                            step.input = f"```json\n{json.dumps(args, indent=2, ensure_ascii=False)}\n```"
                        step.output = f"status={status} | count={data.get('row_count', data.get('count', '-'))}"

                elif event == "token":
                    delta = data.get("delta", "")
                    if not delta:
                        continue
                    if answer_msg is None:
                        answer_msg = cl.Message(content="")
                        await answer_msg.send()
                    await answer_msg.stream_token(delta)

                elif event == "citations":
                    citations_items = data.get("items") or []

                elif event == "hitl_required":
                    hitl_event = data

                elif event == "done":
                    done_meta = data

    # Attach side-panel citation elements and latency footer
    if answer_msg is not None:
        elements: list[cl.Text] = []
        ref_links: list[str] = []
        for idx, item in enumerate(citations_items, start=1):
            label = f"[{idx}] {item.get('ref', 'arXiv')} (p.{item.get('page', '?')})"
            body = (
                f"**{item.get('title', '')}**\n\n"
                f"- **Ref:** {item.get('ref')}\n"
                f"- **Page:** {item.get('page')}\n"
                f"- **Section:** {item.get('section')}\n\n"
                f"> {item.get('snippet', '')}"
            )
            elements.append(cl.Text(name=label, content=body, display="side"))
            ref_links.append(label)

        footer_parts: list[str] = []
        if ref_links:
            footer_parts.append("**Sources:** " + " · ".join(ref_links))
        if done_meta:
            footer_parts.append(
                f"_Latency: {done_meta.get('latency_ms', '-')} ms | "
                f"TTFT: {done_meta.get('ttft_ms', '-')} ms | "
                f"Grounded: {done_meta.get('is_grounded', True)}_"
            )
        if footer_parts:
            await answer_msg.stream_token("\n\n---\n" + "\n".join(footer_parts))

        if elements:
            answer_msg.elements = elements
        await answer_msg.update()

    return hitl_event


async def _handle_hitl(thread_id: str, hitl_data: dict[str, Any]) -> None:
    payload = hitl_data.get("payload") or {}
    action = hitl_data.get("action", "create_reading_note")

    preview = (
        f"### Human-in-the-Loop Approval Required (`{action}`)\n"
        f"```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```\n"
        "Please choose whether to **Approve**, **Edit**, or **Reject** this database write:"
    )

    res = await cl.AskActionMessage(
        content=preview,
        actions=[
            cl.Action(name="approve", payload={"decision": "approve"}, label="✅ Approve"),
            cl.Action(name="edit", payload={"decision": "edit"}, label="✏️ Edit"),
            cl.Action(name="reject", payload={"decision": "reject"}, label="❌ Reject"),
        ],
        timeout=180,
    ).send()

    if not res:
        await _consume_stream(
            "/api/v1/chat/resume",
            {"thread_id": thread_id, "decision": "reject", "reject_reason": "Approval timed out."},
        )
        return

    decision = (res.get("payload") or {}).get("decision") or res.get("name", "reject")

    if decision == "approve":
        await _consume_stream(
            "/api/v1/chat/resume",
            {"thread_id": thread_id, "decision": "approve"},
        )
    elif decision == "edit":
        edit_reply = await cl.AskUserMessage(
            content=(
                "Enter your updated note text (or a JSON object like "
                '`{"note": "...", "priority": "medium", "tags": ["rag"]}`):'
            ),
            timeout=180,
        ).send()
        raw_edit = (edit_reply or {}).get("output", "").strip()
        edited_payload: dict[str, Any] = {}
        if raw_edit.startswith("{"):
            try:
                edited_payload = json.loads(raw_edit)
            except json.JSONDecodeError:
                edited_payload = {"note": raw_edit}
        elif raw_edit:
            edited_payload = {"note": raw_edit}

        await _consume_stream(
            "/api/v1/chat/resume",
            {
                "thread_id": thread_id,
                "decision": "edit",
                "edited_payload": edited_payload,
            },
        )
    else:
        await _consume_stream(
            "/api/v1/chat/resume",
            {
                "thread_id": thread_id,
                "decision": "reject",
                "reject_reason": "Rejected by user in UI.",
            },
        )


@cl.on_chat_start
async def on_chat_start() -> None:
    thread_id = f"ui_{uuid.uuid4().hex[:10]}"
    cl.user_session.set("thread_id", thread_id)
    await cl.Message(
        content=(
            "### AI Research Papers Assistant (Agentic RAG + MCP v3.0)\n"
            "Ask me anything about the **60 indexed arXiv AI papers** (`RAG`, `Agents`, `Fine-tuning`, `Evaluation`), "
            "query benchmark scores in SQLite, compare methodologies with page citations, or save reading notes (with HITL approval)."
        )
    ).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    thread_id = cl.user_session.get("thread_id") or f"ui_{uuid.uuid4().hex[:10]}"
    cl.user_session.set("thread_id", thread_id)

    hitl_data = await _consume_stream(
        "/api/v1/chat/stream",
        {"thread_id": thread_id, "query": message.content},
    )
    if hitl_data is not None:
        await _handle_hitl(thread_id, hitl_data)
