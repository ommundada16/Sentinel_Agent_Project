import pytest
from pydantic import ValidationError

from app.agent.schemas import ALLOWED_TOOLS, TOOL_SPECS, llm_tool_definitions
from app.mcp_server import tools


def test_lookup_and_list(seeded):
    u = tools.lookup_user("Priya Sharma")["matches"][0]
    ents = tools.list_entitlements(u["id"])["entitlements"]
    assert any(e["permission"] == "admin" and e["sensitivity"] == "high" for e in ents)


@pytest.mark.parametrize("uid,eid,allowed", [
    (5, 3, True),    # Finance user is eligible for FinanceApp admin
    (6, 3, False),   # Sales not eligible for FinanceApp admin
    (10, 10, False), # inactive user
    (1, 3, False),   # already holds it
    (6, 1, True),    # low sensitivity, active
])
def test_check_policy_cases(seeded, uid, eid, allowed):
    assert tools.check_policy(uid, eid)["allowed"] is allowed


def test_write_tools(seeded):
    r = tools.request_access(6, 1, "Needs read access for reporting")
    assert r["status"] == "pending"
    assert tools.revoke_access(1, 3, "Role change")["revoked"] is True
    assert "error" in tools.revoke_access(1, 3, "Role change again")


def test_schemas_reject_bad_args():
    spec = TOOL_SPECS["request_access"].args_model
    with pytest.raises(ValidationError):
        spec.model_validate({"user_id": 1, "entitlement_id": 1, "justification": "short"})
    with pytest.raises(ValidationError):
        spec.model_validate({"user_id": 1, "entitlement_id": 1, "justification": "long enough text", "x": 1})
    with pytest.raises(ValidationError):
        TOOL_SPECS["lookup_user"].args_model.model_validate({"name_or_email": 5})


def test_tool_definitions_match_allowlist():
    assert {d["name"] for d in llm_tool_definitions()} == set(ALLOWED_TOOLS) == set(tools.TOOL_FUNCS)
