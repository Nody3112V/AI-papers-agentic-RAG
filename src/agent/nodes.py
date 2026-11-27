"""8 LangGraph Agent Nodes: Contextualizer, Router, RAG/SQL/Hybrid, Batch Grader, Rewriter, HITL, Generator, 2-Tier Checker."""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt

from src.agent.mcp_client import MCPClient
from src.agent.state import (
    AgentState,
    BatchGradeOutput,
    HallucinationResult,
    RouteDecision,
)
from src.core.config import get_prompts, get_settings
from src.core.logging import get_logger
from src.llm.provider import ainvoke_structured, ainvoke_text

logger = get_logger(__name__)

ARXIV_CITATION_RE = re.compile(r"\[arXiv:(\d{4}\.\d{4,5})(?:,\s*p\.\s*(\d+))?\]", re.IGNORECASE)
INSUFFICIENT_EVIDENCE_MSG = (
    "I could not find verified evidence in the indexed AI paper corpus to answer this question."
)


def verify_citations_regex(
    draft_answer: str,
    retrieved_chunks: list[dict[str, Any]],
    sql_results: list[dict[str, Any]] | None = None,
) -> tuple[bool, list[str]]:
    """Tier-1 deterministic citation verifier (0 ms): checks every [arXiv:...] matches context."""
    valid_ids = {str(c.get("arxiv_id", "")).strip() for c in retrieved_chunks if c.get("arxiv_id")}
    if sql_results:
        for row in sql_results:
            if isinstance(row, dict) and row.get("arxiv_id"):
                valid_ids.add(str(row["arxiv_id"]).strip())

    cited_matches = ARXIV_CITATION_RE.findall(draft_answer)
    invalid: list[str] = []
    for cited_id, _page in cited_matches:
        if cited_id not in valid_ids:
            invalid.append(f"Cited arXiv:{cited_id} which is not present in retrieved context.")

    return len(invalid) == 0, invalid


def _format_chunks_for_prompt(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return "(none)"
    blocks: list[str] = []
    for idx, c in enumerate(chunks):
        blocks.append(
            f"[{idx}] arXiv:{c.get('arxiv_id')} | Title: {c.get('title')} | "
            f"Page: {c.get('page_number')} | Section: {c.get('section_path')}\n"
            f"{c.get('text', '')}"
        )
    return "\n\n---\n\n".join(blocks)


def _extract_arxiv_ids_from_sql(sql_results: list[dict[str, Any]] | None) -> list[str]:
    if not sql_results:
        return []
    ids: list[str] = []
    for row in sql_results:
        if isinstance(row, dict) and row.get("arxiv_id"):
            aid = str(row["arxiv_id"]).strip()
            if aid and aid not in ids:
                ids.append(aid)
    return ids


def _build_citations_payload(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for c in chunks:
        aid = str(c.get("arxiv_id", ""))
        page = int(c.get("page_number", 1))
        section = str(c.get("section_path", ""))
        key = (aid, page, section)
        if not aid or key in seen:
            continue
        seen.add(key)
        items.append(
            {
                "ref": f"arXiv:{aid}",
                "title": str(c.get("title", "")),
                "page": page,
                "section": section,
                "snippet": str(c.get("text", ""))[:260],
            }
        )
    return items


def make_nodes(mcp_client: MCPClient) -> dict[str, Any]:
    """Create all 8 agent nodes bound to the provided MCPClient."""

    async def contextualize_query(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages") or []
        original = state.get("original_query") or ""
        if not original and messages:
            last_msg = messages[-1]
            original = getattr(last_msg, "content", str(last_msg))

        # Turn 1 optimization: skip LLM call completely when there is no prior history.
        if len(messages) <= 1:
            return {
                "original_query": original,
                "standalone_query": original,
                "rewritten_query": None,
                "retrieved_chunks": [],
                "sql_results": None,
                "tool_calls_log": [],
                "pending_action": None,
                "retry_count": 0,
                "sql_retry_count": 0,
                "gen_retry_count": 0,
                "citations": [],
                "is_grounded": True,
                "error_trace": [],
            }

        cfg = get_settings()
        prompts = get_prompts()
        window = messages[-(cfg.app.conversation_window_turns * 2) :]
        history_lines: list[str] = []
        for msg in window[:-1]:
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            history_lines.append(f"{role}: {getattr(msg, 'content', str(msg))}")

        prompt_text = (
            f"{prompts['contextualizer']}\n\n"
            f"Conversation history:\n" + "\n".join(history_lines) + f"\n\nLatest user question: {original}"
        )
        standalone = await ainvoke_text(prompt_text, tier="fast")
        return {
            "original_query": original,
            "standalone_query": standalone or original,
            "rewritten_query": None,
            "retrieved_chunks": [],
            "sql_results": None,
            "tool_calls_log": [],
            "pending_action": None,
            "retry_count": 0,
            "sql_retry_count": 0,
            "gen_retry_count": 0,
            "citations": [],
            "is_grounded": True,
            "error_trace": [],
        }

    async def router(state: AgentState) -> dict[str, Any]:
        prompts = get_prompts()
        db_schema, paper_catalog = await asyncio.gather(
            mcp_client.read_resource("db://schema"),
            mcp_client.read_resource("papers://catalog"),
        )
        system_prompt = prompts["router"].format(
            db_schema=db_schema,
            paper_catalog=paper_catalog,
        )
        query = state.get("standalone_query") or state.get("original_query", "")
        decision = await ainvoke_structured(
            RouteDecision,
            [
                ("system", system_prompt),
                ("human", query),
            ],
            tier="fast",
        )
        decision_dict = decision.model_dump()
        pending = decision_dict.get("write_payload") if decision.route == "write_note" else None
        return {
            "route_decision": decision_dict,
            "pending_action": pending,
        }

    async def _run_rag_once(state: AgentState) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        decision = state.get("route_decision") or {}
        query = state.get("rewritten_query") or state.get("standalone_query") or state.get("original_query", "")
        arxiv_ids = decision.get("arxiv_ids")
        if decision.get("route") == "hybrid" and decision.get("hybrid_mode") == "sql_then_rag":
            sql_ids = _extract_arxiv_ids_from_sql(state.get("sql_results"))
            if sql_ids:
                arxiv_ids = sql_ids

        args: dict[str, Any] = {
            "query": query,
            "arxiv_ids": arxiv_ids,
            "year_min": decision.get("year_min"),
            "only_tables": bool(decision.get("only_tables", False)),
            "top_k": get_settings().retrieval.rerank_top_k,
        }
        t0 = time.perf_counter()
        res = await mcp_client.call_tool("search_paper_chunks", args)
        dur = round((time.perf_counter() - t0) * 1000.0, 2)
        chunks = res.get("chunks", []) if res.get("status") == "ok" else []
        log_entry = {
            "tool": "search_paper_chunks",
            "args": args,
            "duration_ms": res.get("duration_ms", dur),
            "status": res.get("status", "ok"),
            "count": len(chunks),
        }
        return chunks, log_entry

    async def _run_sql_once(state: AgentState) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, list[str]]:
        cfg = get_settings()
        decision = state.get("route_decision") or {}
        sql = decision.get("sql_query") or ""
        sql_retries = int(state.get("sql_retry_count", 0))
        errors = list(state.get("error_trace") or [])
        logs: list[dict[str, Any]] = []

        if not sql.strip():
            return [], logs, sql_retries, errors

        t0 = time.perf_counter()
        res = await mcp_client.call_tool("query_papers_db", {"sql": sql})
        dur = round((time.perf_counter() - t0) * 1000.0, 2)
        logs.append(
            {
                "tool": "query_papers_db",
                "args": {"sql": sql},
                "duration_ms": res.get("duration_ms", dur),
                "status": res.get("status", "ok"),
                "row_count": res.get("row_count", 0),
            }
        )

        # Self-repair SQL once if sql_guard / SQLite returned an error
        if res.get("status") != "ok" and sql_retries < cfg.agent.max_sql_retries:
            err_msg = str(res.get("error", "Unknown SQL error"))
            errors.append(f"SQL error: {err_msg}")
            sql_retries += 1
            schema = await mcp_client.read_resource("db://schema")
            repair_prompt = (
                "Fix the following SQLite SELECT query based on the schema and error.\n"
                "Return ONLY one valid read-only SELECT query with LIMIT <= 50, no markdown fences.\n"
                f"<db_schema>{schema}</db_schema>\n"
                f"Broken SQL: {sql}\nError: {err_msg}"
            )
            fixed_sql = await ainvoke_text(repair_prompt, tier="fast")
            fixed_sql = fixed_sql.replace("```sql", "").replace("```", "").strip()
            t1 = time.perf_counter()
            res = await mcp_client.call_tool("query_papers_db", {"sql": fixed_sql})
            dur1 = round((time.perf_counter() - t1) * 1000.0, 2)
            logs.append(
                {
                    "tool": "query_papers_db",
                    "args": {"sql": fixed_sql},
                    "duration_ms": res.get("duration_ms", dur1),
                    "status": res.get("status", "ok"),
                    "row_count": res.get("row_count", 0),
                }
            )

        rows = res.get("rows", []) if res.get("status") == "ok" else []
        return rows, logs, sql_retries, errors

    async def rag_retrieve(state: AgentState) -> dict[str, Any]:
        chunks, log_entry = await _run_rag_once(state)
        logs = list(state.get("tool_calls_log") or []) + [log_entry]
        return {"retrieved_chunks": chunks, "tool_calls_log": logs}

    async def mcp_sql_exec(state: AgentState) -> dict[str, Any]:
        rows, new_logs, sql_retries, errors = await _run_sql_once(state)
        logs = list(state.get("tool_calls_log") or []) + new_logs
        return {
            "sql_results": rows,
            "tool_calls_log": logs,
            "sql_retry_count": sql_retries,
            "error_trace": errors,
        }

    async def hybrid_parallel_exec(state: AgentState) -> dict[str, Any]:
        """Execute SQL and RAG concurrently (latency = max(t_sql, t_rag)) for parallel hybrid mode."""
        (chunks, rag_log), (rows, sql_logs, sql_retries, errors) = await asyncio.gather(
            _run_rag_once(state),
            _run_sql_once(state),
        )
        logs = list(state.get("tool_calls_log") or []) + sql_logs + [rag_log]
        return {
            "retrieved_chunks": chunks,
            "sql_results": rows,
            "tool_calls_log": logs,
            "sql_retry_count": sql_retries,
            "error_trace": errors,
        }

    async def batch_document_grader(state: AgentState) -> dict[str, Any]:
        chunks = state.get("retrieved_chunks") or []
        if not chunks:
            return {"retrieved_chunks": []}

        prompts = get_prompts()
        query = state.get("standalone_query") or state.get("original_query", "")
        numbered = _format_chunks_for_prompt(chunks)
        prompt_text = prompts["batch_grader"].format(query=query, numbered_chunks=numbered)

        grade_out = await ainvoke_structured(BatchGradeOutput, prompt_text, tier="fast")
        relevant_indices = {g.chunk_index for g in grade_out.grades if g.relevant}
        kept = [c for idx, c in enumerate(chunks) if idx in relevant_indices]
        return {"retrieved_chunks": kept}

    async def query_rewriter(state: AgentState) -> dict[str, Any]:
        prompts = get_prompts()
        original = state.get("original_query", "")
        previous = state.get("rewritten_query") or state.get("standalone_query") or original
        prompt_text = prompts["query_rewriter"].format(
            original_query=original,
            previous_query=previous,
        )
        rewritten = await ainvoke_text(prompt_text, tier="fast")
        return {
            "rewritten_query": rewritten or previous,
            "retry_count": int(state.get("retry_count", 0)) + 1,
        }

    async def hitl_write_approval(state: AgentState) -> dict[str, Any]:
        payload = dict(state.get("pending_action") or {})
        decision_input = interrupt(
            {
                "action": "create_reading_note",
                "payload": payload,
            }
        )
        decision_data = decision_input if isinstance(decision_input, dict) else {"decision": str(decision_input)}
        decision = str(decision_data.get("decision", "reject")).lower().strip()

        if decision == "reject":
            reason = decision_data.get("reject_reason") or "Cancelled by user."
            msg = f"Reading note creation was cancelled as requested. ({reason})"
            return {
                "draft_answer": msg,
                "is_grounded": True,
                "pending_action": None,
                "messages": [AIMessage(content=msg)],
            }

        final_payload = dict(payload)
        if decision == "edit" and isinstance(decision_data.get("edited_payload"), dict):
            final_payload.update(decision_data["edited_payload"])

        tool_args = {
            "arxiv_id": str(final_payload.get("arxiv_id", "")),
            "note": str(final_payload.get("note", "")),
            "tags": final_payload.get("tags") or [],
            "priority": str(final_payload.get("priority", "medium")),
        }
        t0 = time.perf_counter()
        res = await mcp_client.call_tool("create_reading_note", tool_args)
        dur = round((time.perf_counter() - t0) * 1000.0, 2)
        logs = list(state.get("tool_calls_log") or []) + [
            {
                "tool": "create_reading_note",
                "args": tool_args,
                "duration_ms": res.get("duration_ms", dur),
                "status": res.get("status", "ok"),
            }
        ]
        if res.get("status") == "ok":
            msg = (
                f"Saved reading note #{res.get('note_id')} for arXiv:{res.get('arxiv_id')} "
                f"({res.get('paper_title')}) with priority `{res.get('priority')}` [SQL: reading_notes]."
            )
        else:
            msg = f"Could not save reading note: {res.get('error', 'Unknown error')}."

        return {
            "draft_answer": msg,
            "is_grounded": True,
            "pending_action": None,
            "tool_calls_log": logs,
            "messages": [AIMessage(content=msg)],
        }

    async def citation_generator(state: AgentState) -> dict[str, Any]:
        decision = state.get("route_decision") or {}
        route = decision.get("route", "rag_only")
        query = state.get("standalone_query") or state.get("original_query", "")
        chunks = state.get("retrieved_chunks") or []
        sql_rows = state.get("sql_results")

        if route == "direct":
            direct_prompt = (
                "You are an AI Research Papers Assistant scoped strictly to an indexed arXiv AI paper corpus. "
                "If the user is greeting you, respond briefly and explain what you can help with. "
                "If the question is outside the AI research paper corpus (e.g., stocks, weather, recipes, "
                "sports, medical/legal advice, or nonexistent papers), politely refuse without fabricating facts.\n\n"
                f"User message: {query}"
            )
            answer = await ainvoke_text(direct_prompt, tier="main")
            return {
                "draft_answer": answer,
                "citations": [],
                "is_grounded": True,
                "messages": [AIMessage(content=answer)],
            }

        # Honest refusal when retrieval + SQL both yielded no verified evidence
        if not chunks and not sql_rows:
            return {
                "draft_answer": INSUFFICIENT_EVIDENCE_MSG,
                "citations": [],
                "is_grounded": True,
                "messages": [AIMessage(content=INSUFFICIENT_EVIDENCE_MSG)],
            }

        prompts = get_prompts()
        sql_str = json.dumps(sql_rows, ensure_ascii=False, indent=2) if sql_rows else "(none)"
        chunks_str = _format_chunks_for_prompt(chunks)
        system_prompt = prompts["citation_generator"].format(
            sql_results=sql_str,
            retrieved_context=chunks_str,
        )

        # If retrying after hallucination check, pass unsupported claims feedback
        extra_feedback = ""
        if int(state.get("gen_retry_count", 0)) > 0 and state.get("error_trace"):
            extra_feedback = (
                "\n\nIMPORTANT CORRECTION: Your previous draft had unsupported claims: "
                + "; ".join(state["error_trace"][-3:])
                + ". Remove any unsupported claims and stick strictly to the provided context."
            )

        answer = await ainvoke_text(
            [
                ("system", system_prompt + extra_feedback),
                ("human", query),
            ],
            tier="main",
        )
        return {
            "draft_answer": answer,
            "citations": _build_citations_payload(chunks),
        }

    async def hallucination_checker(state: AgentState) -> dict[str, Any]:
        cfg = get_settings()
        draft = state.get("draft_answer", "")
        chunks = state.get("retrieved_chunks") or []
        sql_rows = state.get("sql_results")
        errors = list(state.get("error_trace") or [])
        gen_retries = int(state.get("gen_retry_count", 0))

        # Skip check for direct refusals / insufficient evidence responses
        if not draft or draft == INSUFFICIENT_EVIDENCE_MSG:
            return {"is_grounded": True, "messages": [AIMessage(content=draft)]}

        # Tier 1: Deterministic Regex verification (0 ms)
        regex_ok, regex_errors = verify_citations_regex(draft, chunks, sql_rows)
        unsupported: list[str] = list(regex_errors)

        # Tier 2: Optional LLM Judge verification
        llm_grounded = True
        if regex_ok and cfg.agent.enable_llm_hallucination_check:
            prompts = get_prompts()
            evidence = (
                f"SQL Results:\n{json.dumps(sql_rows, ensure_ascii=False) if sql_rows else '(none)'}\n\n"
                f"Paper Chunks:\n{_format_chunks_for_prompt(chunks)}"
            )
            judge_prompt = prompts["hallucination_checker"].format(
                evidence=evidence,
                draft_answer=draft,
            )
            verdict = await ainvoke_structured(HallucinationResult, judge_prompt, tier="fast")
            llm_grounded = verdict.is_grounded
            unsupported.extend(verdict.unsupported_claims)

        is_grounded = regex_ok and llm_grounded
        if is_grounded:
            return {
                "is_grounded": True,
                "messages": [AIMessage(content=draft)],
            }

        # Retry generation once if budget remains
        if gen_retries < cfg.agent.max_gen_retries:
            errors.extend(unsupported or ["Draft answer failed grounding check."])
            return {
                "is_grounded": False,
                "gen_retry_count": gen_retries + 1,
                "error_trace": errors,
            }

        # Budget exhausted -> return honest insufficient-evidence fallback
        return {
            "is_grounded": False,
            "draft_answer": INSUFFICIENT_EVIDENCE_MSG,
            "messages": [AIMessage(content=INSUFFICIENT_EVIDENCE_MSG)],
        }

    return {
        "contextualize_query": contextualize_query,
        "router": router,
        "rag_retrieve": rag_retrieve,
        "mcp_sql_exec": mcp_sql_exec,
        "hybrid_parallel_exec": hybrid_parallel_exec,
        "batch_document_grader": batch_document_grader,
        "query_rewriter": query_rewriter,
        "hitl_write_approval": hitl_write_approval,
        "citation_generator": citation_generator,
        "hallucination_checker": hallucination_checker,
    }
