"""Verify Phase 4 Definition of Done live on Laptop: 5 E2E LangGraph scenarios + RAM check."""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psutil
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agent.graph import AgentGraphRuntime
from src.agent.mcp_client import MCPClient


async def run_scenarios() -> int:
    mcp_client = MCPClient(inprocess=True)
    async with AgentGraphRuntime(mcp_client=mcp_client) as graph:
        scenarios = [
            (
                "1. RAG Single Paper",
                "What are the four types of reflection tokens used in Self-RAG?",
            ),
            (
                "2. MCP SQL Pure",
                "Which model and paper report the highest score on PopQA in our database?",
            ),
            (
                "3. Hybrid (sql_then_rag)",
                "Find the paper with the highest PopQA score in our database and explain the three actions its retrieval evaluator triggers.",
            ),
            (
                "4. Out-of-Domain Refusal",
                "What was the exact closing stock price of NVIDIA on November 1, 2026?",
            ),
        ]

        for label, query in scenarios:
            tid = f"verify_{uuid.uuid4().hex[:8]}"
            cfg = {"configurable": {"thread_id": tid}}
            print(f"\n=== {label} ===")
            out = await graph.ainvoke(
                {"messages": [HumanMessage(content=query)], "thread_id": tid, "original_query": query},
                config=cfg,
            )
            decision = out.get("route_decision") or {}
            tools = [t["tool"] for t in (out.get("tool_calls_log") or [])]
            print(f"Route     : {decision.get('route')} (hybrid_mode={decision.get('hybrid_mode')})")
            print(f"Tools     : {tools}")
            print(f"Grounded  : {out.get('is_grounded')}")
            print(f"Answer    : {str(out.get('draft_answer', ''))[:260]}...")

        # Scenario 5: HITL write_note interrupt() + reject resume (keeps real papers.db clean)
        print("\n=== 5. HITL write_note (interrupt -> reject) ===")
        hitl_tid = f"verify_hitl_{uuid.uuid4().hex[:8]}"
        hitl_cfg = {"configurable": {"thread_id": hitl_tid}}
        hitl_q = (
            "Save a high-priority reading note for arXiv:2310.11511 with tags rag, self-rag: "
            "Review Section 3.2 reflection tokens."
        )
        paused = await graph.ainvoke(
            {"messages": [HumanMessage(content=hitl_q)], "thread_id": hitl_tid, "original_query": hitl_q},
            config=hitl_cfg,
        )
        interrupts = paused.get("__interrupt__")
        print(f"Paused at interrupt(): {bool(interrupts)}")
        if interrupts:
            resumed = await graph.ainvoke(
                Command(resume={"decision": "reject", "reject_reason": "Phase 4 dry-run test."}),
                config=hitl_cfg,
            )
            print(f"Resumed answer       : {resumed.get('draft_answer')}")

    ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"\n[RAM] Current Python process RSS = {ram_mb:.1f} MB (budget < 1200 MB)")
    print("=== PHASE 4 LIVE CHECK COMPLETE ===")
    return 0


def main() -> int:
    return asyncio.run(run_scenarios())


if __name__ == "__main__":
    raise SystemExit(main())
