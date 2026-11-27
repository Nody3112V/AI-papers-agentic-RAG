"""Async MCP client (Streamable HTTP to FastMCP server + in-process fallback for tests/Kaggle)."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)

_RESOURCE_CACHE: dict[str, str] = {}


def clear_resource_cache() -> None:
    _RESOURCE_CACHE.clear()


def _parse_mcp_tool_content(result: Any) -> dict[str, Any]:
    """Extract JSON dict from FastMCP CallToolResult content blocks."""
    content = getattr(result, "content", None) or []
    texts = [getattr(block, "text", "") for block in content if getattr(block, "text", None)]
    joined = "\n".join(texts).strip()
    if not joined:
        return {"status": "ok"}
    try:
        parsed = json.loads(joined)
        return parsed if isinstance(parsed, dict) else {"status": "ok", "data": parsed}
    except json.JSONDecodeError:
        return {"status": "ok", "text": joined}


class MCPClient:
    """Async wrapper over FastMCP Streamable HTTP with clean in-process fallback."""

    def __init__(self, *, server_url: str | None = None, inprocess: bool = False) -> None:
        cfg = get_settings()
        self.server_url = server_url or cfg.env.mcp_server_url
        self.timeout = cfg.mcp.client_timeout_seconds
        self.inprocess = inprocess or (self.server_url.lower() == "inprocess")

    async def read_resource(self, uri: str, *, use_cache: bool = True) -> str:
        if use_cache and uri in _RESOURCE_CACHE:
            return _RESOURCE_CACHE[uri]

        if self.inprocess:
            text = await asyncio.to_thread(self._read_resource_inprocess, uri)
            if use_cache:
                _RESOURCE_CACHE[uri] = text
            return text

        try:
            text = await self._read_resource_http(uri)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MCP HTTP read_resource({uri}) failed ({exc}); using in-process fallback.")
            text = await asyncio.to_thread(self._read_resource_inprocess, uri)

        if use_cache:
            _RESOURCE_CACHE[uri] = text
        return text

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.inprocess:
            return await asyncio.to_thread(self._call_tool_inprocess, name, arguments)

        try:
            return await self._call_tool_http(name, arguments)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MCP HTTP call_tool({name}) failed ({exc}); using in-process fallback.")
            return await asyncio.to_thread(self._call_tool_inprocess, name, arguments)

    async def _read_resource_http(self, uri: str) -> str:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with (
            streamablehttp_client(self.server_url, timeout=self.timeout) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            res = await session.read_resource(uri)
            contents = getattr(res, "contents", []) or []
            return "\n".join(getattr(c, "text", "") for c in contents).strip()

    async def _call_tool_http(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with (
            streamablehttp_client(self.server_url, timeout=self.timeout) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            res = await session.call_tool(name, arguments=arguments)
            return _parse_mcp_tool_content(res)

    @staticmethod
    def _read_resource_inprocess(uri: str) -> str:
        from src.mcp_server import server as mcp_srv

        mapping = {
            "db://schema": mcp_srv.resource_db_schema,
            "papers://catalog": mcp_srv.resource_papers_catalog,
            "papers://stats": mcp_srv.resource_papers_stats,
        }
        if uri not in mapping:
            raise ValueError(f"Unknown MCP resource URI: {uri}")
        return mapping[uri]()

    @staticmethod
    def _call_tool_inprocess(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        from src.mcp_server import server as mcp_srv

        mapping = {
            "query_papers_db": mcp_srv.query_papers_db,
            "search_paper_chunks": mcp_srv.search_paper_chunks,
            "get_paper_metadata": mcp_srv.get_paper_metadata,
            "create_reading_note": mcp_srv.create_reading_note,
        }
        if name not in mapping:
            raise ValueError(f"Unknown MCP tool: {name}")
        return mapping[name](**arguments)
