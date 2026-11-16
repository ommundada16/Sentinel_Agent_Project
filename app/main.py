"""FastAPI layer (spec 5.6). Stateless: all session/approval state lives in the DB."""
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agent.core import Agent, ChatResult, build_agent
from app.data import db
from app.data import repository as repo


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=64)
    message: str


class ApprovalRequest(BaseModel):
    decision: Literal["approve", "deny"]


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    tools_used: list[str]
    pending_approval: dict | None = None


@lru_cache
def get_agent() -> Agent:
    return build_agent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_schema()
    yield


app = FastAPI(title="Sentinel", description="Safe, auditable access-governance agent", lifespan=lifespan)


def _resp(r: ChatResult) -> ChatResponse:
    return ChatResponse(session_id=r.session_id, answer=r.answer, tools_used=r.tools_used,
                        pending_approval=r.pending_approval)


@app.get("/health")
def health():
    return {"status": "ok"}


# WHY plain `def` (not async): the agent loop is blocking I/O; FastAPI runs it in a threadpool.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, agent: Agent = Depends(get_agent)):
    return _resp(agent.chat(req.session_id or uuid.uuid4().hex, req.message))


@app.post("/approvals/{approval_id}", response_model=ChatResponse)
def approve(approval_id: str, req: ApprovalRequest, agent: Agent = Depends(get_agent)):
    try:
        return _resp(agent.resolve_approval(approval_id, req.decision))
    except LookupError:
        raise HTTPException(404, "approval not found or already resolved")


@app.get("/audit")
def audit(session_id: str | None = None, limit: int = 100):
    with db.session_scope() as s:
        return repo.list_audit(s, session_id, limit)
