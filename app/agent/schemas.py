"""Tool schemas: the SINGLE source of truth (spec 7).
The same Pydantic models produce the LLM tool definitions AND validate arguments, so what the
model is told can never drift from what we enforce.
"""
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")  # WHY forbid: reject unexpected/hallucinated args


class LookupUserArgs(_Args):
    name_or_email: str = Field(min_length=2, max_length=200, description="User's name or email")


class ListEntitlementsArgs(_Args):
    user_id: int = Field(gt=0)
    limit: int = Field(default=20, ge=1, le=50)
    offset: int = Field(default=0, ge=0)


class CheckPolicyArgs(_Args):
    user_id: int = Field(gt=0)
    entitlement_id: int = Field(gt=0)


class SearchDocsArgs(_Args):
    query: str = Field(min_length=3, max_length=500)


class RequestAccessArgs(_Args):
    user_id: int = Field(gt=0)
    entitlement_id: int = Field(gt=0)
    justification: str = Field(min_length=10, max_length=500, description="Business reason, >=10 chars")


class RevokeAccessArgs(_Args):
    user_id: int = Field(gt=0)
    entitlement_id: int = Field(gt=0)
    reason: str = Field(min_length=10, max_length=500)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    kind: Literal["READ", "WRITE"]  # drives the approval gate
    description: str
    args_model: type[BaseModel]


TOOL_SPECS: dict[str, ToolSpec] = {s.name: s for s in [
    ToolSpec("lookup_user", "READ", "Find a user by name or email.", LookupUserArgs),
    ToolSpec("list_entitlements", "READ", "List a user's current entitlements with sensitivity.",
             ListEntitlementsArgs),
    ToolSpec("check_policy", "READ", "Deterministically check whether a user may hold an entitlement.",
             CheckPolicyArgs),
    ToolSpec("search_governance_docs", "READ", "Semantic search over governance policy documents.",
             SearchDocsArgs),
    ToolSpec("request_access", "WRITE", "Create an access request (needs human approval).",
             RequestAccessArgs),
    ToolSpec("revoke_access", "WRITE", "Revoke a user's entitlement (needs human approval).",
             RevokeAccessArgs),
]}

# WHY hardcoded allowlist = the keys above; the executor refuses anything else (defense in depth).
ALLOWED_TOOLS = frozenset(TOOL_SPECS)


def llm_tool_definitions() -> list[dict]:
    return [
        {"name": t.name, "description": f"[{t.kind}] {t.description}",
         "input_schema": t.args_model.model_json_schema()}
        for t in TOOL_SPECS.values()
    ]
