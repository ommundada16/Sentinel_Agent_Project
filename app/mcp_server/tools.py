"""Tool implementations. They call the repository only (never raw SQL) and bound their output."""
from app.agent.embeddings import get_embedder
from app.data import db
from app.data import repository as repo

# WHY: access rules are deterministic code, NOT LLM judgment. The model may surface the decision,
# never make it ("recognise when a non-AI solution is better").
HIGH_ALLOWED_DEPARTMENTS = {
    "FinanceApp": {"Finance", "IT", "Security"},
    "HRPortal": {"HR", "IT", "Security"},
}
DEFAULT_HIGH_DEPARTMENTS = {"IT", "Security"}
RELEVANCE_FLOOR = 0.15  # below this the docs don't really answer the question


def lookup_user(name_or_email: str) -> dict:
    with db.session_scope() as s:
        users, more = repo.find_users(s, name_or_email, limit=5)
    return {"matches": users, "more_available": more}


def list_entitlements(user_id: int, limit: int = 20, offset: int = 0) -> dict:
    with db.session_scope() as s:
        if not repo.get_user(s, user_id):
            return {"error": f"user {user_id} not found"}
        items, more = repo.list_entitlements_for_user(s, user_id, limit, offset)
    return {"user_id": user_id, "entitlements": items, "more_available": more}


def check_policy(user_id: int, entitlement_id: int) -> dict:
    with db.session_scope() as s:
        user = repo.get_user(s, user_id)
        ent = repo.get_entitlement(s, entitlement_id)
        if not user or not ent:
            return {"allowed": False, "reason": "unknown user or entitlement"}
        if user["status"] != "active":
            return {"allowed": False, "reason": "user is inactive; inactive users are never eligible"}
        if repo.user_has_entitlement(s, user_id, entitlement_id):
            return {"allowed": False, "reason": "user already holds this entitlement"}
    if ent["sensitivity"] == "high":
        allowed_depts = HIGH_ALLOWED_DEPARTMENTS.get(ent["app_name"], DEFAULT_HIGH_DEPARTMENTS)
        if user["department"] not in allowed_depts:
            return {"allowed": False,
                    "reason": f"high-sensitivity access to {ent['app_name']} is limited to "
                              f"{sorted(allowed_depts)}; user is in {user['department']}"}
    return {"allowed": True, "reason": "meets policy (active user, eligible department)"}


def search_governance_docs(query: str) -> dict:
    vec = get_embedder().embed([query])[0]  # SAME embedder as ingestion
    with db.session_scope() as s:
        hits = repo.search_doc_chunks(s, vec, k=4)
    hits = [h for h in hits if h["score"] >= RELEVANCE_FLOOR]
    if not hits:
        return {"results": [], "note": "No relevant policy documents found."}
    return {"results": hits}


def request_access(user_id: int, entitlement_id: int, justification: str) -> dict:
    with db.session_scope() as s:
        if not repo.get_user(s, user_id) or not repo.get_entitlement(s, entitlement_id):
            return {"error": "unknown user or entitlement"}
        return repo.create_access_request(s, user_id, entitlement_id, justification)


def revoke_access(user_id: int, entitlement_id: int, reason: str) -> dict:
    with db.session_scope() as s:
        if not repo.user_has_entitlement(s, user_id, entitlement_id):
            return {"error": "user does not hold this entitlement"}
        return repo.revoke_entitlement(s, user_id, entitlement_id, reason)


TOOL_FUNCS = {f.__name__: f for f in [
    lookup_user, list_entitlements, check_policy, search_governance_docs, request_access, revoke_access,
]}
