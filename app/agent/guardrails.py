"""Guardrail layer (spec 5.4). Each check is a small, separately testable function."""
import json
import re

from app.agent.schemas import ALLOWED_TOOLS

MAX_INPUT_CHARS = 2000
MAX_TOOL_RESULT_CHARS = 4000  # WHY: unbounded tool output blows the context window (spec 5.7)


class GuardrailError(ValueError):
    pass


# 1. Input validation ---------------------------------------------------------------
def validate_input(text: str) -> str:
    # WHY strip control chars: they can hide instructions from human reviewers.
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text or "").strip()
    if not cleaned:
        raise GuardrailError("empty message")
    if len(cleaned) > MAX_INPUT_CHARS:
        raise GuardrailError(f"message too long (max {MAX_INPUT_CHARS} characters)")
    return cleaned


# 2. Tool allowlist -----------------------------------------------------------------
def is_allowed_tool(name: str) -> bool:
    # WHY: models can hallucinate tool names; only names we hardcoded may ever run.
    return name in ALLOWED_TOOLS


# 4. Prompt-injection defense ------------------------------------------------------
INJECTION_PATTERNS = {
    "ignore_instructions": r"ignore\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?)",
    "disregard": r"disregard\s+.{0,40}(instructions?|rules?|polic(y|ies))",
    "role_change": r"\byou\s+are\s+now\b|\bact\s+as\s+(an?\s+)?(admin|root|system|developer)",
    "new_instructions": r"\bnew\s+instructions?\s*:",
    "fake_system_msg": r"\b(system|developer)\s+(prompt|message|override)\s*:",
    "embedded_tool_request": r"\b(call|invoke|run|execute|use)\s+(the\s+)?(revoke_access|request_access|"
                             r"check_policy|lookup_user|list_entitlements|search_governance_docs)\b",
    "address_the_ai": r"\b(assistant|ai agent|language model)\s*[,:]\s*(you\s+)?(must|should|please|immediately)\b",
    "secrecy": r"\bdo\s+not\s+(tell|inform|notify|alert)\s+the\s+user\b",
    "exfiltration": r"\b(reveal|print|show|leak)\s+.{0,30}(system\s+prompt|api\s+key|secret|password)",
}
_COMPILED = {k: re.compile(v, re.I) for k, v in INJECTION_PATTERNS.items()}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
OPEN_TAG, CLOSE_TAG = "<untrusted_data", "</untrusted_data>"


def detect_injection(text: str) -> list[str]:
    return [name for name, rx in _COMPILED.items() if rx.search(text)]


def _clean_string(text: str, flags: list[str]) -> str:
    kept = []
    for sentence in _SENTENCE_SPLIT.split(text):
        hit = detect_injection(sentence)
        if hit:
            flags.extend(hit)
            kept.append("[REMOVED: suspected prompt injection]")  # quarantine, don't obey
        else:
            kept.append(sentence)
    return " ".join(kept)


def _clean(obj, flags: list[str]):
    if isinstance(obj, str):
        return _clean_string(obj, flags)
    if isinstance(obj, dict):
        return {k: _clean(v, flags) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v, flags) for v in obj]
    return obj


def sanitize_untrusted(result: dict, source: str) -> tuple[str, list[str]]:
    """Turn a tool result into delimited, bounded, injection-quarantined text for the model.
    WHY: tool/doc output is DATA. We strip instruction-like sentences, bound the size, and wrap
    it in delimiters the system prompt tells the model never to obey."""
    flags: list[str] = []
    text = json.dumps(_clean(result, flags), default=str)
    text = text.replace(CLOSE_TAG, "[delimiter removed]")  # block delimiter break-out
    if len(text) > MAX_TOOL_RESULT_CHARS:
        text = text[:MAX_TOOL_RESULT_CHARS] + "...[truncated]"
    return f'{OPEN_TAG} source="{source}">\n{text}\n{CLOSE_TAG}', sorted(set(flags))


# 5. Approval gate ------------------------------------------------------------------
def requires_approval(kind: str, sensitivity: str | None) -> bool:
    # WHY fail closed: anything that writes, or touches a high-sensitivity entitlement, needs a human.
    return kind == "WRITE" or sensitivity == "high"


# 6. Output validation + audit redaction ---------------------------------------------
_SECRETS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}"), re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), re.compile(r"\b(?:\d[ -]?){13,16}\b"),
]
_EMAIL = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def redact_secrets(text: str) -> str:
    for rx in _SECRETS:
        text = rx.sub("[REDACTED]", text)
    return text


def validate_output(answer: str, tools_used: list[str]) -> str:
    answer = redact_secrets(answer or "")
    if tools_used:  # traceability: the user can see which tools informed the answer
        answer += f"\n\n(Tools used: {', '.join(dict.fromkeys(tools_used))})"
    return answer


def redact_for_audit(obj):
    """WHY: audit logs must not hold secrets or full PII verbatim. Emails are masked."""
    if isinstance(obj, str):
        return _EMAIL.sub(r"\1***@\2", redact_secrets(obj))
    if isinstance(obj, dict):
        return {k: redact_for_audit(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact_for_audit(v) for v in obj]
    return obj
