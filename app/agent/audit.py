"""Structured audit logging (spec 5.5). Every turn and tool call becomes a DB row + a JSON log line."""
import json
import logging

from app.agent.guardrails import redact_for_audit
from app.config import get_settings
from app.data import db
from app.data import repository as repo

logger = logging.getLogger("sentinel.audit")


class AuditLogger:
    def __init__(self, session_id: str):
        self.session_id = session_id

    def log(self, actor: str, action: str, tool: str | None = None, args: dict | None = None,
            result: dict | None = None, approval_status: str | None = None, latency_ms: float = 0.0,
            prompt_tokens: int = 0, completion_tokens: int = 0, notes: str = "") -> None:
        cfg = get_settings()
        cost = (prompt_tokens * cfg.price_in_per_m + completion_tokens * cfg.price_out_per_m) / 1_000_000
        fields = dict(
            session_id=self.session_id, actor=actor, action=action, tool=tool,
            args_json=redact_for_audit(args) if args is not None else None,
            result_json=redact_for_audit(result) if result is not None else None,
            approval_status=approval_status, latency_ms=round(latency_ms, 2),
            tokens=prompt_tokens + completion_tokens, cost_usd=cost, notes=notes,
        )
        # WHY own transaction: the audit row is committed even if the surrounding work later fails.
        with db.session_scope() as s:
            repo.add_audit(s, **fields)
        logger.info(json.dumps({k: v for k, v in fields.items() if v not in (None, "", 0, 0.0)}, default=str))
