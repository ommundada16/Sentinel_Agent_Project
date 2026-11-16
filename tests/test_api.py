from fastapi.testclient import TestClient

from app.agent.core import Agent
from app.agent.providers import FakeLLM
from app.data import db
from app.data import repository as repo
from app.main import app, get_agent
from app.mcp_server.client import LocalExecutor


def client_with(script):
    agent = Agent(FakeLLM(script), LocalExecutor())  # ONE agent so the script advances across requests
    app.dependency_overrides[get_agent] = lambda: agent
    return TestClient(app)  # no `with`: skips lifespan, the test fixture already built the schema


def test_health_and_chat_read(seeded):
    c = client_with([{"tools": [{"name": "lookup_user", "args": {"name_or_email": "Priya Sharma"}}]},
                     {"text": "Found her."}])
    assert c.get("/health").json() == {"status": "ok"}
    body = c.post("/chat", json={"session_id": "a1", "message": "who is Priya Sharma?"}).json()
    assert body["tools_used"] == ["lookup_user"] and body["pending_approval"] is None


def test_approval_flow_and_audit(seeded):
    c = client_with([
        {"tools": [{"name": "revoke_access", "args": {"user_id": 1, "entitlement_id": 3, "reason": "Role change"}}]},
        {"text": "Revoked."},
    ])
    r = c.post("/chat", json={"session_id": "a2", "message": "revoke it"}).json()
    approval_id = r["pending_approval"]["id"]
    done = c.post(f"/approvals/{approval_id}", json={"decision": "approve"}).json()
    assert "Revoked." in done["answer"]
    with db.session_scope() as s:
        assert not repo.user_has_entitlement(s, 1, 3)
    assert c.post(f"/approvals/{approval_id}", json={"decision": "approve"}).status_code == 404
    assert c.post(f"/approvals/{approval_id}", json={"decision": "maybe"}).status_code == 422
    actions = [a["action"] for a in c.get("/audit", params={"session_id": "a2"}).json()]
    assert {"approval_requested", "tool_call", "final_answer"} <= set(actions)


def teardown_module(_):
    app.dependency_overrides.clear()
