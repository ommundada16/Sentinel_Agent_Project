"""SQLAlchemy models (spec 5.1)."""
from datetime import UTC, datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

EMBED_DIM = 384  # all-MiniLM-L6-v2 output size


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), index=True)
    email: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    department: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(20), default="active")  # active | inactive


class Role(Base):
    __tablename__ = "roles"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    description: Mapped[str] = mapped_column(String(300), default="")


class Entitlement(Base):
    __tablename__ = "entitlements"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    app_name: Mapped[str] = mapped_column(String(100), index=True)
    permission: Mapped[str] = mapped_column(String(100))
    sensitivity: Mapped[str] = mapped_column(String(10))  # low | medium | high


class UserRole(Base):
    __tablename__ = "user_roles"
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), primary_key=True)
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), primary_key=True, index=True)


class RoleEntitlement(Base):
    __tablename__ = "role_entitlements"
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"), primary_key=True)
    entitlement_id: Mapped[int] = mapped_column(ForeignKey("entitlements.id"), primary_key=True, index=True)


class AccessRequest(Base):
    __tablename__ = "access_requests"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    entitlement_id: Mapped[int] = mapped_column(ForeignKey("entitlements.id"), index=True)
    status: Mapped[str] = mapped_column(String(20))  # pending | approved | denied | revoked
    requested_by: Mapped[str] = mapped_column(String(100), default="agent")
    decided_by: Mapped[str | None] = mapped_column(String(100), nullable=True)
    justification: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True, default="")
    actor: Mapped[str] = mapped_column(String(50))
    action: Mapped[str] = mapped_column(String(50))
    tool: Mapped[str | None] = mapped_column(String(50), nullable=True)
    args_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    approval_status: Mapped[str | None] = mapped_column(String(20), nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    notes: Mapped[str] = mapped_column(Text, default="")


class DocChunk(Base):
    __tablename__ = "doc_chunks"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(100), index=True)
    doc_title: Mapped[str] = mapped_column(String(200))
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    # WHY variant: real pgvector column on Postgres; JSON list on SQLite (dev/test fallback only).
    embedding = mapped_column(Vector(EMBED_DIM).with_variant(JSON, "sqlite"))
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)


class ChatSession(Base):
    """Conversation state lives in the DB, not process memory => API stays stateless (5.7)."""

    __tablename__ = "chat_sessions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    messages_json: Mapped[list] = mapped_column(JSON, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class PendingApproval(Base):
    __tablename__ = "pending_approvals"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    tool_call_json: Mapped[dict] = mapped_column(JSON)  # {id, name, args}
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending|approved|denied
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
