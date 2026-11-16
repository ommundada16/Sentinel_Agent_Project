"""LLM providers (Strategy pattern, spec 4). The agent only knows `complete()`.

Internal message format (provider-neutral, JSON-serialisable so it can live in the DB):
  {"role":"user","content":str}
  {"role":"assistant","content":str,"tool_calls":[{"id","name","args"}]}
  {"role":"tool","tool_call_id":str,"name":str,"content":str}
"""
import itertools
from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMProvider(Protocol):
    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse: ...


class FakeLLM:
    """Scripted responses for deterministic tests/evals. WHY: free, fast, tests the ORCHESTRATION.
    Script items: {"text": "..."} or {"tools": [{"name":..., "args":{...}}]}."""

    _ids = itertools.count(1)

    def __init__(self, script: list[dict]):
        self.script = list(script)
        self.calls: list[list[dict]] = []  # messages seen on each call, for assertions

    def complete(self, system, messages, tools) -> LLMResponse:
        self.calls.append(list(messages))
        if not self.script:
            return LLMResponse(text="(fake llm: script exhausted)", prompt_tokens=10, completion_tokens=5)
        step = self.script.pop(0)
        calls = [ToolCall(f"call_{next(self._ids)}", t["name"], t.get("args", {})) for t in step.get("tools", [])]
        return LLMResponse(text=step.get("text", ""), tool_calls=calls, prompt_tokens=100, completion_tokens=20)


class AnthropicProvider:
    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    @staticmethod
    def _convert(messages: list[dict]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["content"]})
            elif m["role"] == "assistant":
                blocks = [{"type": "text", "text": m["content"]}] if m["content"] else []
                blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["args"]}
                           for c in m.get("tool_calls", [])]
                out.append({"role": "assistant", "content": blocks})
            else:  # tool results must be grouped into ONE user message
                block = {"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def complete(self, system, messages, tools) -> LLMResponse:
        r = self.client.messages.create(model=self.model, max_tokens=1024, system=system,
                                        messages=self._convert(messages), tools=tools)
        text = "".join(b.text for b in r.content if b.type == "text")
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in r.content if b.type == "tool_use"]
        return LLMResponse(text, calls, r.usage.input_tokens, r.usage.output_tokens)


class OllamaProvider:
    """Local, free option. Uses Ollama's /api/chat with tool calling."""

    def __init__(self, url: str, model: str):
        self.url, self.model = url.rstrip("/"), model

    def complete(self, system, messages, tools) -> LLMResponse:
        import httpx

        msgs = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "assistant":
                msg = {"role": "assistant", "content": m["content"]}
                if m.get("tool_calls"):
                    msg["tool_calls"] = [{"function": {"name": c["name"], "arguments": c["args"]}}
                                         for c in m["tool_calls"]]
                msgs.append(msg)
            elif m["role"] == "tool":
                msgs.append({"role": "tool", "content": m["content"], "tool_name": m["name"]})
            else:
                msgs.append(m)
        fn_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                      "parameters": t["input_schema"]}} for t in tools]
        r = httpx.post(f"{self.url}/api/chat", timeout=120,
                       json={"model": self.model, "messages": msgs, "tools": fn_tools, "stream": False})
        r.raise_for_status()
        data = r.json()
        msg = data["message"]
        calls = [ToolCall(f"call_{next(FakeLLM._ids)}", c["function"]["name"], c["function"]["arguments"])
                 for c in msg.get("tool_calls", [])]
        return LLMResponse(msg.get("content", ""), calls, data.get("prompt_eval_count", 0), data.get("eval_count", 0))


def get_llm(settings) -> LLMProvider:
    if settings.llm_provider == "ollama":
        return OllamaProvider(settings.ollama_url, settings.ollama_model)
    return AnthropicProvider(settings.anthropic_api_key, settings.anthropic_model)
