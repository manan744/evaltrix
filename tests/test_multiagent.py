from __future__ import annotations

import pytest

from evaltrix import FakeJev, Trace, evaluate
from evaltrix.evals import ToolRisk
from evaltrix.trace import units

MULTI = {
    "trace_id": "tr_multi",
    "agents": [
        {"name": "orchestrator", "description": "Routes requests", "tools": []},
        {"name": "billing", "description": "Refunds and invoices", "tools": ["lookup_order", "issue_refund"]},
    ],
    "messages": [
        {"role": "user", "content": "Order C901 arrived broken, please refund it."},
        {"role": "assistant", "agent": "orchestrator", "content": "Routing you to billing."},
        {"role": "handoff", "from": "orchestrator", "to": "billing",
         "content": "Refund order C901 for the customer; it arrived damaged."},
        {"role": "assistant", "agent": "billing", "tool_calls": [
            {"id": "b1", "function": {"name": "lookup_order", "arguments": '{"order_id": "C901"}'}}]},
        {"role": "tool", "tool_call_id": "b1", "content": '{"order_id": "C901", "total": 120}'},
        {"role": "assistant", "agent": "billing", "tool_calls": [
            {"id": "b2", "function": {"name": "issue_refund", "arguments": '{"order_id": "C901", "amount": 120}'}}]},
        {"role": "tool", "tool_call_id": "b2", "content": '{"refund_id": "rf_1", "status": "issued"}'},
        {"role": "handoff", "from": "billing", "to": "orchestrator", "content": "Refunded $120 on C901 (rf_1)."},
        {"role": "assistant", "agent": "orchestrator", "content": "Your refund of $120 is on its way."},
    ],
}


def test_agents_handoffs_and_attribution():
    t = Trace.from_any(MULTI)
    assert t.is_multi_agent
    assert t.agent_names == ["orchestrator", "billing"]
    assert [(h.source, h.target, h.returning) for h in t.handoffs] == [
        ("orchestrator", "billing", False), ("billing", "orchestrator", True)]
    assert t.root_agent == "orchestrator" and t.delegated_agents == ["billing"]
    assert [(c.name, c.agent) for c in t.tool_calls] == [("lookup_order", "billing"), ("issue_refund", "billing")]
    assert {m.agent for m in t.messages if m.role == "tool"} == {"billing"}  # results inherit the caller
    assert t.task_of("billing") == "Refund order C901 for the customer; it arrived damaged."
    assert t.task_of("orchestrator") == t.first_user_message  # the top-level agent works for the user
    assert t.allowed_tools("billing") == ["lookup_order", "issue_refund"]
    assert t.allowed_tools("orchestrator") == []
    assert t.final_answer == "Your refund of $120 is on its way."


def test_single_agent_traces_are_unaffected():
    t = Trace.from_any([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}])
    assert not t.is_multi_agent and t.handoffs == [] and t.task_of(None) == "hi"
    assert [u.key for u in units(t)] == ["trace"]


def test_units_carry_agent_and_delegated_task():
    t = Trace.from_any(MULTI)
    by_key = {u.key: u for u in units(t)}
    assert list(by_key) == [
        "trace",
        "handoff[0]:orchestrator->billing",
        "tool_call[0]:lookup_order", "tool_result[0]:lookup_order",
        "tool_call[1]:issue_refund", "tool_result[1]:issue_refund",
        "handoff[1]:billing->orchestrator",
        "agent:billing",
    ]
    refund = by_key["tool_call[1]:issue_refund"]
    assert refund.agent == "billing" and refund.task.startswith("Refund order C901")

    h = by_key["handoff[0]:orchestrator->billing"]
    assert h.agent == "orchestrator" and h.task == t.first_user_message and h.handoff.task.startswith("Refund")

    a = by_key["agent:billing"]
    assert a.task.startswith("Refund order C901")
    assert [m.role for m in a.trace.messages] == ["handoff", "assistant", "tool", "assistant", "tool", "handoff"]
    assert all(m.agent in ("billing", "orchestrator") for m in a.trace.messages)
    assert "user" not in {m.role for m in a.trace.messages}  # a sub-agent sees its task, not the user


async def test_tool_risk_judges_a_subagent_call_against_its_delegated_task():
    fake = FakeJev(fixed={"tool_risk.action": "approve"})
    await evaluate(MULTI, [ToolRisk()], jev=fake)
    refund_state = next(s for s, _ in fake.calls if s["proposed_call"]["tool"] == "issue_refund")
    assert refund_state["task"] == "Refund order C901 for the customer; it arrived damaged."


def test_handoff_needs_a_target():
    with pytest.raises(ValueError, match="needs a `to`"):
        Trace.from_any([{"role": "handoff", "from": "a", "content": "x"}])


async def test_session_context_is_visible_to_tool_call_rules():
    t = Trace.from_any({"context": {"signed_in_customer": "cus_7"}, "messages": [
        {"role": "user", "content": "show my orders"},
        {"role": "assistant", "tool_calls": [{"id": "1", "function": {"name": "list_orders",
                                                                     "arguments": '{"customer_id": "cus_7"}'}}]}]})
    assert t.context == {"signed_in_customer": "cus_7"} and "context" not in t.metadata
    fake = FakeJev(fixed={"tool_risk.action": "approve"})
    await evaluate(t, [ToolRisk()], jev=fake)
    assert fake.calls[0][0]["session_context"] == {"signed_in_customer": "cus_7"}


async def test_delegation_scope_sees_reports_from_its_own_subagents():
    from evaltrix.evals import DelegationScope

    chain = {"messages": [
        {"role": "user", "content": "cheapest flight?"},
        {"role": "handoff", "from": "orchestrator", "to": "planner", "content": "find the cheapest flight"},
        {"role": "handoff", "from": "planner", "to": "flights", "content": "search and return the cheapest"},
        {"role": "assistant", "agent": "flights", "tool_calls": [{"id": "f", "function": {"name": "search", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "f", "content": "AA1 $148"},
        {"role": "handoff", "from": "flights", "to": "planner", "content": "AA1 at $148"},
        {"role": "handoff", "from": "planner", "to": "orchestrator", "content": "AA1, $148"},
        {"role": "assistant", "agent": "orchestrator", "content": "AA1 for $148."}]}
    fake = FakeJev()
    await evaluate(chain, [DelegationScope()], jev=fake)
    planner = next(s for s, _ in fake.calls if s["agent"] == "planner")
    assert planner["received_reports"] == [{"from": "flights", "content": "AA1 at $148"}]
    assert planner["actions"] == [] and planner["report_back"] == ["AA1, $148"]
