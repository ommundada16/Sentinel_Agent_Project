"""ALL database access lives here (spec 5.1).

WHY: one auditable place; SQLAlchemy parameterises every query (no SQL injection surface);
every read is filtered + LIMITed so it works the same at 15 rows or 15M (spec 5.7).
"""
import math

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.orm import Session

from app.data.models import (
    AccessRequest, AuditLog, ChatSession, DocChunk, Entitlement, PendingApproval,
    RoleEntitlement, User, UserRole,
)

MAX_ROWS = 50


def _row(obj, *fields):
    return {f: getattr(obj, f) for f in fields}


USER_FIELDS = ("id", "name", "email", "department", "status")
ENT_FIELDS = ("id", "app_name", "permission", "sensitivity")


# ---------- users / entitlements ----------
def find_users(s: Session, name_or_email: str, limit: int = 5) -> tuple[list[dict], bool]:
    q = name_or_email.strip().lower()
    stmt = (
        select(User)
        .where(or_(func.lower(User.email) == q, func.lower(User.name).like(f"%{q}%")))
        .order_by(User.id)
        .limit(min(limit, MAX_ROWS) + 1)  # +1 to learn whether more rows exist
    )
    rows = list(s.scalars(stmt))
    more = len(rows) > limit
    return [_row(u, *USER_FIELDS) for u in rows[:limit]], more


def get_user(s: Session, user_id: int) -> dict | None:
    u = s.get(User, user_id)
    return _row(u, *USER_FIELDS) if u else None


def get_entitlement(s: Session, entitlement_id: int) -> dict | None:
    e = s.get(Entitlement, entitlement_id)
    return _row(e, *ENT_FIELDS) if e else None


def _revoked_ids(user_id: int):
    return select(AccessRequest.entitlement_id).where(
        and_(AccessRequest.user_id == user_id, AccessRequest.status == "revoked")
    )


def list_entitlements_for_user(s: Session, user_id: int, limit: int = 20, offset: int = 0):
    """Effective entitlements = via roles OR approved requests, minus revoked ones."""
    via_roles = (
        select(RoleEntitlement.entitlement_id)
        .join(UserRole, UserRole.role_id == RoleEntitlement.role_id)
        .where(UserRole.user_id == user_id)
    )
    via_requests = select(AccessRequest.entitlement_id).where(
        and_(AccessRequest.user_id == user_id, AccessRequest.status == "approved")
    )
    stmt = (
        select(Entitlement)
        .where(Entitlement.id.in_(via_roles.union(via_requests)), Entitlement.id.not_in(_revoked_ids(user_id)))
        .order_by(Entitlement.id)
        .limit(min(limit, MAX_ROWS) + 1)
        .offset(offset)
    )
    rows = list(s.scalars(stmt))
    more = len(rows) > limit
    return [_row(e, *ENT_FIELDS) for e in rows[:limit]], more


def user_has_entitlement(s: Session, user_id: int, entitlement_id: int) -> bool:
    items, _ = list_entitlements_for_user(s, user_id, limit=MAX_ROWS)
    return any(e["id"] == entitlement_id for e in items)


# ---------- writes ----------
def create_access_request(s: Session, user_id: int, entitlement_id: int, justification: str,
                          requested_by: str = "agent") -> dict:
    r = AccessRequest(user_id=user_id, entitlement_id=entitlement_id, status="pending",
                      requested_by=requested_by, justification=justification)
    s.add(r)
    s.flush()
    return {"request_id": r.id, "status": r.status}


def revoke_entitlement(s: Session, user_id: int, entitlement_id: int, reason: str,
                       decided_by: str = "human-approver") -> dict:
    # WHY a 'revoked' row instead of deleting role links: role membership is shared by many users.
    s.add(AccessRequest(user_id=user_id, entitlement_id=entitlement_id, status="revoked",
                        requested_by="agent", decided_by=decided_by, justification=reason))
    s.flush()
    return {"revoked": True, "user_id": user_id, "entitlement_id": entitlement_id}


# ---------- vector search ----------
def search_doc_chunks(s: Session, query_vec: list[float], k: int = 4) -> list[dict]:
    k = min(k, 10)
    if s.get_bind().dialect.name == "postgresql":
        dist = DocChunk.embedding.cosine_distance(query_vec)  # pgvector <=> operator, HNSW-indexed
        stmt = select(DocChunk, (1 - dist).label("score")).order_by(dist).limit(k)
        return [
            {"doc_id": c.doc_id, "title": c.doc_title, "chunk_index": c.chunk_index,
             "content": c.content, "score": round(float(sc), 4)}
            for c, sc in s.execute(stmt)
        ]
    # SQLite fallback (dev/tests only): brute-force cosine in Python. Postgres path is the real one.
    scored = []
    for c in s.scalars(select(DocChunk)):
        a, b = query_vec, c.embedding
        dot = sum(x * y for x, y in zip(a, b))
        den = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))) or 1.0
        scored.append((dot / den, c))
    scored.sort(key=lambda t: -t[0])
    return [
        {"doc_id": c.doc_id, "title": c.doc_title, "chunk_index": c.chunk_index,
         "content": c.content, "score": round(sc, 4)}
        for sc, c in scored[:k]
    ]


def add_doc_chunks(s: Session, rows: list[dict]) -> None:
    s.add_all(DocChunk(**r) for r in rows)


def clear_doc_chunks(s: Session) -> None:
    s.execute(delete(DocChunk))


# ---------- sessions / approvals / audit ----------
def load_messages(s: Session, session_id: str) -> list[dict]:
    row = s.get(ChatSession, session_id)
    return list(row.messages_json) if row else []


def save_messages(s: Session, session_id: str, messages: list[dict]) -> None:
    row = s.get(ChatSession, session_id)
    if row:
        row.messages_json = messages
    else:
        s.add(ChatSession(id=session_id, messages_json=messages))


def create_pending_approval(s: Session, approval_id: str, session_id: str, tool_call: dict) -> None:
    s.add(PendingApproval(id=approval_id, session_id=session_id, tool_call_json=tool_call))


def get_pending_approval(s: Session, approval_id: str) -> PendingApproval | None:
    return s.get(PendingApproval, approval_id)


def add_audit(s: Session, **fields) -> None:
    s.add(AuditLog(**fields))


def list_audit(s: Session, session_id: str | None = None, limit: int = 100) -> list[dict]:
    stmt = select(AuditLog).order_by(AuditLog.id.desc()).limit(min(limit, 500))
    if session_id:
        stmt = stmt.where(AuditLog.session_id == session_id)
    return [
        {**_row(a, "id", "session_id", "actor", "action", "tool", "args_json", "result_json",
                "approval_status", "latency_ms", "tokens", "cost_usd", "notes"), "ts": a.ts.isoformat()}
        for a in s.scalars(stmt)
    ]
