"""The one trace shape every eval sees, whatever framework produced the log.

    t = Trace.from_any(raw)       # OpenAI messages, Anthropic content blocks, or {"input", "output"}
    t.user_messages, t.final_answer, t.tool_calls, t.tool_results, t.system_prompt, t.tools

Messages are normalized to:
    {"role": "system"|"user"|"assistant"|"tool"|"handoff", "content": str,
     "agent": str | None              # which agent produced it (assistant, tool, handoff)
     "tool_calls": [ToolCall, ...]    # assistant only
     "tool_call_id": str              # tool only
     "to": str                        # handoff only: the agent receiving the task in `content`}

Multi-agent traces mark each assistant message with its `agent` and record delegation as
`{"role": "handoff", "from": "orchestrator", "to": "billing", "content": "<the delegated task>"}`.
A handoff back (`to` the delegating agent) carries the sub-agent's report. Single-agent traces
need none of this: `agent` stays None and everything works as before.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str | None = None
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None  # filled in from the matching tool message, if any
    agent: str | None = None  # the agent that made the call


class Message(BaseModel):
    role: str
    content: str = ""
    agent: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    to: str | None = None  # handoff target


class Tool(BaseModel):
    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    """What an agent in the trace is, as declared by the system that ran it (optional)."""

    name: str
    description: str = ""
    tools: list[str] | None = None  # tool names it is allowed to call; None = not declared


class Handoff(BaseModel):
    index: int  # position in trace.messages
    source: str | None
    target: str
    task: str  # the delegated task, or the report for a return
    returning: bool = False  # True when `target` had earlier delegated to `source`: a report back, not a new task


class Trace(BaseModel):
    messages: list[Message]
    tools: list[Tool] = Field(default_factory=list)
    agents: list[AgentSpec] = Field(default_factory=list)
    # facts the agent legitimately has outside the messages: signed-in user, tenant, locale, session ids.
    # Tool-call rules treat argument values found here as grounded.
    context: dict[str, Any] = Field(default_factory=dict)
    id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)  # everything else from the raw log (timestamp, version...)

    @classmethod
    def from_any(cls, raw: Any) -> Trace:
        from .normalize import to_trace

        return to_trace(raw)

    # ------------------------------------------------------------------ derived views

    @property
    def system_prompt(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "system" and m.content)

    @property
    def user_messages(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "user" and m.content]

    @property
    def first_user_message(self) -> str:
        return self.user_messages[0] if self.user_messages else ""

    @property
    def final_answer(self) -> str:
        """The last assistant text, if the trace ends with (or contains) one."""
        for m in reversed(self.messages):
            if m.role == "assistant" and m.content:
                return m.content
        return ""

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Every tool call in order, each paired with its result.

        Results are matched by id; a tool message without a usable id goes to the oldest call still
        waiting for a result (some frameworks don't log ids).
        """
        calls: list[ToolCall] = []
        by_id: dict[str, ToolCall] = {}
        waiting: list[ToolCall] = []
        for m in self.messages:
            if m.role == "assistant":
                for tc in m.tool_calls:
                    c = tc.model_copy()
                    calls.append(c)
                    waiting.append(c)
                    if c.id:
                        by_id[c.id] = c
            elif m.role == "tool":
                c = by_id.get(m.tool_call_id or "")
                if c is None or c not in waiting:
                    c = waiting[0] if waiting else None
                if c is not None:
                    c.result = m.content
                    waiting.remove(c)
        return calls

    @property
    def tool_results(self) -> list[str]:
        return [m.content for m in self.messages if m.role == "tool"]

    # ------------------------------------------------------------------ multi-agent

    @property
    def is_multi_agent(self) -> bool:
        return bool(self.handoffs) or len({m.agent for m in self.messages if m.agent}) > 1

    @property
    def handoffs(self) -> list[Handoff]:
        out: list[Handoff] = []
        for i, m in enumerate(self.messages):
            if m.role == "handoff" and m.to:
                returning = any(h.source == m.to and h.target == m.agent and not h.returning for h in out)
                out.append(Handoff(index=i, source=m.agent, target=m.to, task=m.content, returning=returning))
        return out

    @property
    def root_agent(self) -> str | None:
        """The top-level agent: the first one to act. None for traces without agent names."""
        return next((m.agent for m in self.messages if m.role in ("assistant", "handoff") and m.agent), None)

    @property
    def delegated_agents(self) -> list[str]:
        """Agents that received a task from another agent, in order of first delegation."""
        root = self.root_agent
        return list(dict.fromkeys(h.target for h in self.handoffs if not h.returning and h.target != root))

    @property
    def agent_names(self) -> list[str]:
        seen: list[str] = []
        for name in [a.name for a in self.agents] + [m.agent for m in self.messages] + [h.target for h in self.handoffs]:
            if name and name not in seen:
                seen.append(name)
        return seen

    def agent_spec(self, name: str | None) -> AgentSpec | None:
        return next((a for a in self.agents if a.name == name), None)

    def task_of(self, agent: str | None, before: int | None = None) -> str:
        """What `agent` was asked to do as of message `before`: the latest task handed to it,
        or the user's request for the top-level agent (and for single-agent traces)."""
        limit = len(self.messages) if before is None else before
        for h in reversed(self.handoffs):
            if h.index < limit and h.target == agent and agent is not None and not h.returning:
                return h.task
        return self.first_user_message

    def allowed_tools(self, agent: str | None) -> list[str] | None:
        """Tools this agent may call, if declared: per-agent list first, else the trace's tool list."""
        spec = self.agent_spec(agent)
        if spec and spec.tools is not None:
            return spec.tools
        return [t.name for t in self.tools] if self.tools else None

    def upto(self, n_messages: int) -> Trace:
        """The same trace cut after its first `n_messages` messages: what the agent had seen at that point."""
        return self.model_copy(update={"messages": self.messages[:n_messages]})

    def for_agent(self, agent: str) -> Trace:
        """Only what `agent` did and saw: tasks handed to it, its own messages and calls, their results,
        and the reports it handed back."""
        call_owner = {c.id: c.agent for c in self.tool_calls if c.id}
        keep = []
        for m in self.messages:
            if m.role == "handoff" and agent in (m.to, m.agent):
                keep.append(m)
            elif m.role == "assistant" and m.agent == agent:
                keep.append(m)
            elif m.role == "tool" and (m.agent == agent or call_owner.get(m.tool_call_id) == agent):
                keep.append(m)
        return self.model_copy(update={"messages": keep})
