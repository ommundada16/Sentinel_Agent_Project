"""Tool executors. The agent depends only on `.call(name, args) -> dict`.

LocalExecutor: in-process (tests/evals, fast). MCPExecutor: real MCP client over stdio.
"""
import asyncio
import json
import os
import sys
import threading
from typing import Protocol

from app.mcp_server.tools import TOOL_FUNCS


class ToolExecutor(Protocol):
    def call(self, name: str, args: dict) -> dict: ...


class LocalExecutor:
    def call(self, name: str, args: dict) -> dict:
        return TOOL_FUNCS[name](**args)


class MCPExecutor:
    """Keeps ONE stdio session open on a background event loop (spawning a server per call is slow).
    The session lives inside a single long-lived task so it is opened and closed by the same task."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()
        self._lock = threading.Lock()
        self._session = None
        self._stop: asyncio.Event | None = None
        self._task = None

    async def _serve(self, started: asyncio.Future) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        # WHY pass env explicitly: the MCP SDK only forwards a minimal default env, but the server
        # needs DATABASE_URL etc. to reach the same database as the agent.
        params = StdioServerParameters(command=sys.executable, args=["-m", "app.mcp_server.server"],
                                       env=dict(os.environ))
        try:
            async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                self._session = session
                started.set_result(None)
                await self._stop.wait()
        except Exception as e:
            if not started.done():
                started.set_exception(e)

    async def _start(self) -> None:
        self._stop = asyncio.Event()
        started = asyncio.get_running_loop().create_future()
        self._task = asyncio.create_task(self._serve(started))
        await started

    def _ensure(self) -> None:
        with self._lock:
            if self._session is None:
                asyncio.run_coroutine_threadsafe(self._start(), self._loop).result(timeout=60)

    async def _call(self, name: str, args: dict) -> dict:
        res = await self._session.call_tool(name, args)
        if res.isError:
            raise RuntimeError(res.content[0].text if res.content else "tool error")
        if getattr(res, "structuredContent", None):
            sc = res.structuredContent
            return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
        return json.loads(res.content[0].text)

    def call(self, name: str, args: dict) -> dict:
        self._ensure()
        return asyncio.run_coroutine_threadsafe(self._call(name, args), self._loop).result(timeout=60)

    def list_tools(self) -> list[str]:
        self._ensure()

        async def _list():
            return [t.name for t in (await self._session.list_tools()).tools]

        return asyncio.run_coroutine_threadsafe(_list(), self._loop).result(timeout=60)

    def close(self) -> None:
        if self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
            if self._task:
                asyncio.run_coroutine_threadsafe(asyncio.wait({self._task}, timeout=10), self._loop).result(15)
