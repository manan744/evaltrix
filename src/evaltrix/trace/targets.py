"""Split a trace into the units evals are judged on.

    "trace"        the whole conversation
    "tool_call"    one call, with the conversation as it stood when the agent proposed it
    "tool_result"  one tool output, with the conversation up to and including it
    "agent"        one delegated agent's share of a multi-agent trace (see Trace.for_agent),
                   judged against the task it was handed
    "handoff"      one delegation from one agent to another

Every unit knows which `agent` it belongs to and that agent's `task`: the user's request for the
top-level agent, or the latest task handed to a sub-agent. A sub-agent's tool call is therefore
judged against what it was delegated, not against the user's whole request.

Judging tool results one by one matters: an injected instruction in the second of five tool outputs
is invisible to a check that only looks at the trace's last tool result.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .model import Handoff, ToolCall, Trace

Target = Literal["trace", "tool_call", "tool_result", "agent", "handoff"]
TARGETS: tuple[Target, ...] = ("trace", "tool_call", "tool_result", "agent", "handoff")


@dataclass(frozen=True)
class Unit:
    kind: Target
    trace: Trace  # cut or filtered to what is relevant for this unit
    index: int = 0  # position among units of the same kind
    agent: str | None = None
    task: str = ""  # what `agent` was asked to do
    call: ToolCall | None = None  # tool_call: the call; tool_result: the call that produced it
    result: str | None = None  # tool_result only
    handoff: Handoff | None = None  # handoff only
    full: Trace | None = None  # the whole trace, when `trace` is a cut or a filter

    @property
    def key(self) -> str:
        """Stable label for reports and audit records: "trace", "tool_call[1]:issue_refund", "agent:billing"."""
        if self.kind == "trace":
            return "trace"
        if self.kind == "agent":
            return f"agent:{self.agent}"
        if self.kind == "handoff":
            h = self.handoff
            return f"handoff[{self.index}]:{h.source if h else '?'}->{h.target if h else '?'}"
        return f"{self.kind}[{self.index}]:{self.call.name if self.call else '?'}"


def units(trace: Trace, kinds: set[Target] | None = None) -> list[Unit]:
    kinds = set(TARGETS) if kinds is None else kinds
    out: list[Unit] = []
    if "trace" in kinds:
        out.append(Unit("trace", trace, task=trace.first_user_message))

    paired = trace.tool_calls  # calls in order, with results attached
    n_call = n_result = n_handoff = 0
    for i, m in enumerate(trace.messages):
        if m.role == "assistant":
            for _ in m.tool_calls:
                call = paired[n_call]
                if "tool_call" in kinds:
                    out.append(Unit("tool_call", trace.upto(i), n_call, agent=call.agent,
                                    task=trace.task_of(call.agent, i), call=call, full=trace))
                n_call += 1
        elif m.role == "tool":
            if "tool_result" in kinds:
                call = next((c for c in paired if c.id and c.id == m.tool_call_id), None)
                if call is None:  # no id to match: the n-th result belongs to the n-th call
                    call = paired[n_result] if n_result < len(paired) else None
                agent = call.agent if call else m.agent
                out.append(Unit("tool_result", trace.upto(i + 1), n_result, agent=agent,
                                task=trace.task_of(agent, i), call=call, result=m.content, full=trace))
            n_result += 1
        elif m.role == "handoff":
            if "handoff" in kinds:
                h = next(h for h in trace.handoffs if h.index == i)
                out.append(Unit("handoff", trace.upto(i + 1), n_handoff, agent=h.source,
                                task=trace.task_of(h.source, i), handoff=h, full=trace))
            n_handoff += 1

    if "agent" in kinds:
        for name in trace.delegated_agents:
            out.append(Unit("agent", trace.for_agent(name), agent=name, task=trace.task_of(name), full=trace))
    return out

