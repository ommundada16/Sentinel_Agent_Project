"""Tool executors. The agent depends only on `.call(name, args) -> dict`.

LocalExecutor: in-process (tests/evals, fast). MCPExecutor: real MCP client over stdio.
"""
import asyncio
import json
import sys
import threading
from contextlib import AsyncExitStack
from typing import Protocol

from app.mcp_server.tools import TOOL_FUNCS


class ToolExecutor(Protocol):
    def call(self, name: str, args: dict) -> dict: ...


class LocalExecutor:
    def call(self, name: str, args: dict) -> dict:
        return TOOL_FUNCS[name](**args)


class MCPExecutor:
    """Keeps ONE stdio session open on a background event loop (spawning a server per call is slow)."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._stack = AsyncExitStack()
        self._session = None

    async def _connect(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server.server"])
        read, write = await self._stack.enter_async_context(stdio_client(params))
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()

    async def _call(self, name: str, args: dict) -> dict:
        if self._session is None:
            await self._connect()
        res = await self._session.call_tool(name, args)
        if res.isError:
            raise RuntimeError(res.content[0].text if res.content else "tool error")
        if getattr(res, "structuredContent", None):
            sc = res.structuredContent
            return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
        return json.loads(res.content[0].text)

    def call(self, name: str, args: dict) -> dict:
        return asyncio.run_coroutine_threadsafe(self._call(name, args), self._loop).result(timeout=60)

    def list_tools(self) -> list[str]:
        async def _list():
            if self._session is None:
                await self._connect()
            return [t.name for t in (await self._session.list_tools()).tools]

        return asyncio.run_coroutine_threadsafe(_list(), self._loop).result(timeout=60)
