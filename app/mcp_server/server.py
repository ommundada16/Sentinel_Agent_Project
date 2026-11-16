"""MCP server exposing the six tools over stdio. Run: python -m app.mcp_server.server
WHY MCP: standard, decoupled tool layer; a reviewer can audit tools without reading the agent.
"""
from mcp.server.fastmcp import FastMCP

from app.mcp_server import tools

mcp = FastMCP("sentinel-tools")


@mcp.tool(description="[READ] Find a user by name or email.")
def lookup_user(name_or_email: str) -> dict:
    return tools.lookup_user(name_or_email)


@mcp.tool(description="[READ] List a user's current entitlements with sensitivity.")
def list_entitlements(user_id: int, limit: int = 20, offset: int = 0) -> dict:
    return tools.list_entitlements(user_id, limit, offset)


@mcp.tool(description="[READ] Deterministically check whether a user may hold an entitlement.")
def check_policy(user_id: int, entitlement_id: int) -> dict:
    return tools.check_policy(user_id, entitlement_id)


@mcp.tool(description="[READ] Semantic search over governance policy documents.")
def search_governance_docs(query: str) -> dict:
    return tools.search_governance_docs(query)


@mcp.tool(description="[WRITE] Create an access request. The agent gates this behind human approval.")
def request_access(user_id: int, entitlement_id: int, justification: str) -> dict:
    return tools.request_access(user_id, entitlement_id, justification)


@mcp.tool(description="[WRITE] Revoke a user's entitlement. The agent gates this behind human approval.")
def revoke_access(user_id: int, entitlement_id: int, reason: str) -> dict:
    return tools.revoke_access(user_id, entitlement_id, reason)


if __name__ == "__main__":
    mcp.run(transport="stdio")
