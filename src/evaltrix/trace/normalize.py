"""Raw agent logs -> Trace.

Accepted inputs:
  - a Trace (returned as is)
  - a list of messages
  - a dict with "messages" (and optionally "tools", "agents", "context", "system", "id"/"trace_id";
    anything else -> metadata). "context" holds session facts the agent knows outside the messages.
  - a dict with "input" / "output" (single-turn, no tools)

Message formats, mixed freely:
  - OpenAI chat: tool_calls [{id, function: {name, arguments: json-string}}], role "tool" with tool_call_id,
    legacy function_call / role "function", content as a string or a list of {"type": "text"} parts
  - Anthropic: content blocks text / tool_use {id, name, input} / tool_result {tool_use_id, content},
    tool results arriving inside a user message, system prompt as a top-level "system" field

Multi-agent: an assistant message names its agent with "agent" (or OpenAI's "name"); delegation is
{"role": "handoff", "from": <agent>, "to": <agent>, "content": <task or report>}. Tool calls and tool
results inherit the agent of the message that made the call.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .model import AgentSpec, Message, Tool, ToolCall, Trace

_ID_KEYS = ("trace_id", "id")
_STRUCTURAL = {"messages", "tools", "agents", "context", "system", "input", "output", *_ID_KEYS}


def to_trace(raw: Any) -> Trace:
    if isinstance(raw, Trace):
        return raw
    if isinstance(raw, (list, tuple)):
        return Trace(messages=normalize_messages(raw))
    if not isinstance(raw, Mapping):
        raise TypeError(f"can't build a trace from {type(raw).__name__}")

    messages: list[Message] = []
    if raw.get("system"):
        messages.append(Message(role="system", content=text_of(raw["system"])))
    if raw.get("messages") is not None:
        messages += normalize_messages(raw["messages"])
    else:
        if raw.get("input") is not None:
            messages.append(Message(role="user", content=text_of(raw["input"])))
        if raw.get("output") is not None:
            messages.append(Message(role="assistant", content=text_of(raw["output"])))
    if not messages:
        raise ValueError("trace has no messages (expected `messages`, or `input`/`output`)")

    trace_id = next((str(raw[k]) for k in _ID_KEYS if raw.get(k) is not None), None)
    return Trace(
        messages=messages,
        tools=normalize_tools(raw.get("tools")),
        context=dict(raw.get("context") or {}),
        agents=[AgentSpec(**a) if isinstance(a, Mapping) else AgentSpec(name=str(a)) for a in raw.get("agents") or []],
        id=trace_id,
        metadata={k: v for k, v in raw.items() if k not in _STRUCTURAL},
    )


def text_of(content: Any) -> str:
    """Flatten message content to text. Non-text structures (tool output dicts) become JSON."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, Mapping) and p.get("type") in ("text", "input_text", "output_text"):
                parts.append(str(p.get("text", "")))
            elif isinstance(p, str):
                parts.append(p)
            elif isinstance(p, Mapping) and p.get("type") in ("tool_use", "tool_result"):
                continue  # handled structurally by the caller
            else:
                parts.append(json.dumps(p, default=str))
        return "\n".join(x for x in parts if x)
    if isinstance(content, Mapping):
        return json.dumps(content, default=str)
    return str(content)


def parse_args(raw: Any) -> dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, str):
        try:
            v = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return v if isinstance(v, dict) else {"_value": v}
    return {"_value": raw}


def normalize_messages(raw: Any) -> list[Message]:
    out: list[Message] = []
    for m in raw:
        if not isinstance(m, Mapping):
            raise TypeError(f"message must be a mapping, got {type(m).__name__}")
        role = m.get("role", "user")
        content = m.get("content")
        agent = m.get("agent") or (m.get("name") if role == "assistant" else None)

        if role == "handoff":
            if not m.get("to"):
                raise ValueError("a handoff message needs a `to` agent")
            out.append(Message(role="handoff", agent=m.get("from") or agent, to=m["to"], content=text_of(content)))
            continue

        # Anthropic: tool_result blocks inside a user turn become tool messages, in order
        if isinstance(content, list):
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "tool_result":
                    out.append(Message(role="tool", tool_call_id=block.get("tool_use_id"),
                                       content=text_of(block.get("content"))))

        calls: list[ToolCall] = []
        for tc in m.get("tool_calls") or []:  # OpenAI
            fn = tc.get("function") or {}
            calls.append(ToolCall(id=tc.get("id"), name=tc.get("name") or fn.get("name") or "", agent=agent,
                                  args=parse_args(tc.get("args", tc.get("arguments", fn.get("arguments"))))))
        if m.get("function_call"):  # OpenAI legacy
            fc = m["function_call"]
            calls.append(ToolCall(name=fc.get("name", ""), agent=agent, args=parse_args(fc.get("arguments"))))
        if isinstance(content, list):  # Anthropic tool_use blocks
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "tool_use":
                    calls.append(ToolCall(id=block.get("id"), name=block.get("name", ""), agent=agent,
                                          args=parse_args(block.get("input"))))

        text = text_of(content)
        if role in ("tool", "function"):
            out.append(Message(role="tool", content=text, tool_call_id=m.get("tool_call_id"), agent=m.get("agent")))
        elif text or calls or role != "user":
            # a user turn that only carried tool_result blocks has nothing left to add
            out.append(Message(role=role, content=text, tool_calls=calls,
                               agent=agent if role == "assistant" else None))
    return _attribute_tool_results(out)


def _attribute_tool_results(messages: list[Message]) -> list[Message]:
    """A tool result belongs to the agent whose call produced it (matched by id, else oldest pending call)."""
    owner: dict[str, str | None] = {}
    pending: list[str | None] = []
    for m in messages:
        if m.role == "assistant":
            for c in m.tool_calls:
                pending.append(c.agent)
                if c.id:
                    owner[c.id] = c.agent
        elif m.role == "tool":
            agent = owner.get(m.tool_call_id or "", pending[0] if pending else None)
            if pending:
                pending.pop(0)
            if m.agent is None:
                m.agent = agent
    return messages


def normalize_tools(raw: Any) -> list[Tool]:
    if not raw:
        return []
    out = []
    for t in raw:
        if isinstance(t, str):
            out.append(Tool(name=t))
            continue
        fn = t.get("function") if t.get("type") == "function" and isinstance(t.get("function"), Mapping) else t
        out.append(Tool(name=fn.get("name", ""), description=fn.get("description") or "",
                        parameters=fn.get("parameters") or fn.get("input_schema") or {}))
    return out
