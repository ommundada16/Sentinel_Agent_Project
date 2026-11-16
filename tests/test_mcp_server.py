"""Smoke test of the REAL MCP stdio path: spawns the server as a subprocess."""
import os

import pytest

from app.data import db
from app.data.seed import seed
from app.mcp_server.client import MCPExecutor


@pytest.fixture()
def file_db(tmp_path, monkeypatch):
    url = f"sqlite:///{tmp_path / 'mcp.db'}"
    monkeypatch.setenv("DATABASE_URL", url)  # inherited by the server subprocess
    monkeypatch.setenv("EMBEDDING_PROVIDER", "hash")
    from app.config import get_settings

    get_settings.cache_clear()
    saved = (db._engine, db._factory)  # restore the shared test engine afterwards
    eng = db.configure(url)
    db.init_schema(eng)
    with db.session_scope() as s:
        seed(s)
    yield
    eng.dispose()
    db._engine, db._factory = saved


@pytest.mark.skipif(os.environ.get("TEST_DATABASE_URL", "").startswith("postgres"), reason="sqlite file db only")
def test_mcp_roundtrip(file_db):
    ex = MCPExecutor()
    assert set(ex.list_tools()) == {"lookup_user", "list_entitlements", "check_policy",
                                    "search_governance_docs", "request_access", "revoke_access"}
    users = ex.call("lookup_user", {"name_or_email": "Priya Sharma"})
    assert users["matches"][0]["id"] == 1
