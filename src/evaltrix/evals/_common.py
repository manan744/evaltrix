"""State builders shared by the built-in rules.

Rules that describe the trace under the same key with the same value get packed into one Jev
request, so these helpers are the single place each shared view is built:

    "task"           what the judged agent was asked to do
    "actions"        tool calls in order (agent, tool, args, result) - agent only in multi-agent traces
    "final_answer"   the last assistant text
    "steps"          the whole run in order: user, agent messages, tool calls/results, handoffs
"""

from __future__ import annotations

import re
from typing import Any

from ..text import truncate
from ..trace.model import Trace
from ..trace.targets import Unit

RESULT_CHARS = 1500

_ERROR = re.compile(
    r'"(error|errors|exception)"\s*:|"status"\s*:\s*"?(error|failed|failure|4\d\d|5\d\d)\b|\btraceback\b|'
    r"^\s*(error|exception|failed|failure)\b|\b(timed out|timeout|permission denied|unauthorized|forbidden|"
    r"not found|rate limit(ed)?|internal server error)\b",
    re.I | re.M,
)


def looks_like_error(text: str | None) -> bool:
    """Cheap first pass for failed tool calls. Jev confirms; this only decides whether to ask."""
    return bool(text) and bool(_ERROR.search(text))  # type: ignore[arg-type]


def actions(t: Trace, result_chars: int = RESULT_CHARS) -> list[dict[str, Any]]:
    multi = t.is_multi_agent
    out = []
    for c in t.tool_calls:
        a: dict[str, Any] = {"tool": c.name, "args": c.args,
                             "result": truncate(c.result, result_chars) if c.result is not None else None}
        if multi:
            a = {"agent": c.agent, **a}
        out.append(a)
    return out


def steps(t: Trace, chars: int = 800) -> list[dict[str, Any]]:
    multi = t.is_multi_agent
    out: list[dict[str, Any]] = []
    for m in t.messages:
        if m.role == "system":
            continue
        if m.role == "handoff":
            out.append({"handoff": {"from": m.agent, "to": m.to, "content": truncate(m.content, chars)}})
            continue
        s: dict[str, Any] = {"role": m.role}
        if multi and m.agent:
            s["agent"] = m.agent
        if m.content:
            s["content"] = truncate(m.content, chars)
        if m.tool_calls:
            s["tool_calls"] = [{"tool": c.name, "args": c.args} for c in m.tool_calls]
        out.append(s)
    return out


def agent_text(t: Trace) -> str:
    """Everything the agents said (not tool output), in order."""
    return "\n".join(m.content for m in t.messages if m.role == "assistant" and m.content)


def task(u: Unit) -> str:
    return u.task or u.trace.first_user_message


def session_context(u: Unit) -> dict[str, Any]:
    return (u.full or u.trace).context


def proposed_call(u: Unit) -> dict[str, Any]:
    assert u.call is not None
    tool = next((t for t in u.trace.tools if t.name == u.call.name), None)
    return {"tool": u.call.name, "description": tool.description if tool else "", "args": u.call.args}


def recent_conversation(u: Unit, n: int = 4) -> list[dict[str, Any]]:
    out = []
    for m in u.trace.messages[-n:]:
        if m.role == "system":
            continue
        d: dict[str, Any] = {"role": m.role, "content": truncate(m.content, RESULT_CHARS)}
        if m.agent:
            d["agent"] = m.agent
        if m.tool_calls:
            d["tool_calls"] = [{"name": c.name, "args": c.args} for c in m.tool_calls]
        out.append(d)
    return out
