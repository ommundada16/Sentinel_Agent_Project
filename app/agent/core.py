"""The agent loop (spec 5.3): explicit, bounded, no framework.

validate input -> LLM -> for each tool call: allowlist -> schema -> approval gate -> execute via MCP
-> sanitize result as untrusted data -> audit -> feed back -> repeat (max_steps) -> validated answer.
"""
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import ValidationError

from app.agent import guardrails as g
from app.agent.audit import AuditLogger
from app.agent.providers import LLMProvider, LLMResponse, ToolCall
from app.agent.schemas import TOOL_SPECS, llm_tool_definitions
from app.config import Settings, get_settings
from app.data import db
from app.data import repository as repo
from app.mcp_server.client import ToolExecutor

SYSTEM_PROMPT = (Path(__file__).parent.parent / "prompts" / "system.txt").read_text(encoding="utf-8")


@dataclass
class ChatResult:
    session_id: str
    answer: str
    tools_used: list[str] = field(default_factory=list)
    pending_approval: dict | None = None


def _tool_msg(call_id: str, name: str, content: str) -> dict:
    return {"role": "tool", "tool_call_id": call_id, "name": name, "content": content}


class Agent:
    def __init__(self, llm: LLMProvider, executor: ToolExecutor, settings: Settings | None = None):
        self.llm, self.executor = llm, executor
        self.max_steps = (settings or get_settings()).max_steps

    # ---- public API -------------------------------------------------------------
    def chat(self, session_id: str, message: str) -> ChatResult:
        audit = AuditLogger(session_id)
        try:
            clean = g.validate_input(message)
        except g.GuardrailError as e:
            audit.log("guardrail", "input_rejected", notes=str(e))
            return ChatResult(session_id, f"Request rejected: {e}.")
        with db.session_scope() as s:
            open_id = repo.open_approval_for_session(s, session_id)
            messages = repo.load_messages(s, session_id)
        if open_id:  # WHY: a dangling tool call would corrupt the conversation; make the human decide first
            return ChatResult(session_id, f"Please approve or deny pending action {open_id} first.")
        messages.append({"role": "user", "content": clean})
        audit.log("user", "user_query", args={"query": clean})
        return self._run(session_id, messages, audit, [])

    def resolve_approval(self, approval_id: str, decision: str) -> ChatResult:
        # WHY default-deny: anything other than an explicit "approve" is a denial.
        status = "approved" if decision == "approve" else "denied"
        with db.session_scope() as s:
            info = repo.resolve_pending_approval(s, approval_id, status)
            if not info:
                raise LookupError("approval not found or already resolved")
            messages = repo.load_messages(s, info["session_id"])
        session_id, call = info["session_id"], info["tool_call"]
        audit = AuditLogger(session_id)
        tools_used: list[str] = []
        if status == "approved":
            content = self._execute(call["name"], call["args"], call["id"], audit, tools_used, "approved")
        else:
            audit.log("human", "approval_denied", tool=call["name"], args=call["args"], approval_status="denied")
            content = "The human reviewer DENIED this action. It was not executed."
        messages.append(_tool_msg(call["id"], call["name"], content))
        return self._run(session_id, messages, audit, tools_used)

    # ---- the loop ---------------------------------------------------------------
    def _run(self, session_id: str, messages: list[dict], audit: AuditLogger, tools_used: list[str]) -> ChatResult:
        for step in range(self.max_steps):  # WHY bounded: no runaway/expensive tool loops
            t0 = time.perf_counter()
            resp: LLMResponse = self.llm.complete(SYSTEM_PROMPT, messages, llm_tool_definitions())
            audit.log("agent", "llm_call", latency_ms=(time.perf_counter() - t0) * 1000,
                      prompt_tokens=resp.prompt_tokens, completion_tokens=resp.completion_tokens,
                      notes=f"step {step + 1}")
            messages.append({"role": "assistant", "content": resp.text,
                             "tool_calls": [{"id": c.id, "name": c.name, "args": c.args} for c in resp.tool_calls]})
            if not resp.tool_calls:
                answer = g.validate_output(resp.text, tools_used)
                audit.log("agent", "final_answer", result={"answer": answer})
                self._save(session_id, messages)
                return ChatResult(session_id, answer, tools_used)

            pending: dict | None = None
            for call in resp.tool_calls:
                if pending:
                    messages.append(_tool_msg(call.id, call.name, "Not executed: another action awaits human approval."))
                    continue
                content, pending = self._handle_call(session_id, call, audit, tools_used)
                if not pending:
                    messages.append(_tool_msg(call.id, call.name, content))
            if pending:
                self._save(session_id, messages)
                return ChatResult(session_id, f"Human approval required to run {pending['tool']}. "
                                  f"Approval id: {pending['id']}", tools_used, pending)

        audit.log("guardrail", "max_steps_exceeded", notes=f"stopped after {self.max_steps} steps")
        safe = "I couldn't complete this request within the allowed number of steps."
        messages.append({"role": "assistant", "content": safe, "tool_calls": []})
        self._save(session_id, messages)
        return ChatResult(session_id, safe, tools_used)

    def _handle_call(self, session_id: str, call: ToolCall, audit: AuditLogger, tools_used: list[str]):
        """Returns (tool_message_content, pending_approval_or_None). Errors go BACK to the model
        so it can self-correct, instead of crashing the loop."""
        if not g.is_allowed_tool(call.name):
            audit.log("guardrail", "blocked_tool", tool=call.name, args=call.args, notes="not in allowlist")
            return f"Error: unknown tool '{call.name}'. Use only the provided tools.", None
        spec = TOOL_SPECS[call.name]
        try:
            args = spec.args_model.model_validate(call.args).model_dump()
        except ValidationError as e:
            problems = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
            audit.log("guardrail", "schema_rejected", tool=call.name, args=call.args, notes=problems)
            return f"Error: invalid arguments ({problems}). Fix them or ask the user.", None

        sensitivity = None
        if "entitlement_id" in args:
            with db.session_scope() as s:
                ent = repo.get_entitlement(s, args["entitlement_id"])
            sensitivity = ent["sensitivity"] if ent else None
        if g.requires_approval(spec.kind, sensitivity):
            approval_id = uuid.uuid4().hex
            tool_call = {"id": call.id, "name": call.name, "args": args}
            with db.session_scope() as s:
                repo.create_pending_approval(s, approval_id, session_id, tool_call)
            audit.log("agent", "approval_requested", tool=call.name, args=args, approval_status="pending")
            return "", {"id": approval_id, "tool": call.name, "args": args}
        return self._execute(call.name, args, call.id, audit, tools_used, None), None

    def _execute(self, name: str, args: dict, call_id: str, audit: AuditLogger, tools_used: list[str],
                 approval_status: str | None) -> str:
        t0 = time.perf_counter()
        try:
            result = self.executor.call(name, args)
        except Exception as e:  # tool failure is reported to the model, not raised
            audit.log("agent", "tool_error", tool=name, args=args, approval_status=approval_status, notes=repr(e))
            return f"Tool error: {type(e).__name__}"
        tools_used.append(name)
        content, flags = g.sanitize_untrusted(result, f"tool:{name}")
        audit.log("agent", "tool_call", tool=name, args=args, result=result, approval_status=approval_status,
                  latency_ms=(time.perf_counter() - t0) * 1000,
                  notes=f"injection_flags={flags}" if flags else "")
        return content

    @staticmethod
    def _save(session_id: str, messages: list[dict]) -> None:
        with db.session_scope() as s:
            repo.save_messages(s, session_id, messages)


def build_agent(settings: Settings | None = None) -> Agent:
    from app.agent.providers import get_llm
    from app.mcp_server.client import LocalExecutor, MCPExecutor

    settings = settings or get_settings()
    executor = MCPExecutor() if settings.tool_transport == "mcp" else LocalExecutor()
    return Agent(get_llm(settings), executor, settings)
