from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from evaltrix import FakeJev
from evaltrix.evals import ALL_RULES
from evaltrix.governance import AuditLog, PolicyError, PolicyPack, observe, render

from .fixtures.agent_scenarios import SCENARIOS

EXAMPLE = Path(__file__).parents[1] / "examples" / "policies" / "agents.yaml"
MIN = {"pack": "t", "version": "1"}


def _traces(*names):
    out = []
    for s in SCENARIOS:
        if not names or s["name"] in names:
            out.append({"trace_id": s["name"], "timestamp": "2026-10-03T10:00:00Z", "agent_version": "a@1",
                        "secret_note": "do not record me", **s["trace"]})
    return out


# --------------------------------------------------------------------------- policy


def test_every_rule_exposes_its_arguments():
    """Policies read back a rule's effective arguments for the digest; that only works if rules keep them."""
    for cls in ALL_RULES:
        rule = cls()
        params = [p for p in inspect.signature(cls.__init__).parameters if p not in ("self", "args", "kwargs", "kw")]
        assert all(hasattr(rule, p) for p in params), cls.name


def test_example_policy_loads():
    pack = PolicyPack.load(EXAMPLE)
    names = [r.name for r in pack.rules]
    assert {"goal_hijacking", "oversight_bypass", "prompt_injection"} <= set(names)
    assert pack.config("prompt_injection").severity == "low"
    assert pack.config("tool_risk").owner == "platform-team"
    assert pack.config("grounded").owner == "ai-governance@example.com"  # inherits the pack owner
    assert pack.config("run_stats").rule.max_tool_calls == 20
    assert len(pack.digest) == 64 and pack.fail_on == "high"


def test_overrides_keep_profile_settings_and_disable_rules():
    strict = PolicyPack.from_dict({**MIN, "profile": "strict", "rules": {"tool_risk": {"args": {"allow_at": 0.95}},
                                                                         "loop_detection": {"enabled": False}}})
    tr = strict.config("tool_risk").rule
    assert tr.allow_at == 0.95 and tr.block_at == 0.4  # user override + strict profile's own value
    assert "loop_detection" not in [r.name for r in strict.rules]
    only = PolicyPack.from_dict({**MIN, "profile": "none", "rules": {"pii": {}}})
    assert [r.name for r in only.rules] == ["pii"]


@pytest.mark.parametrize("spec, match", [
    ({"version": "1"}, "`pack` is required"),
    ({**MIN, "rules": {"nope": {}}}, "unknown rule 'nope'"),
    ({**MIN, "rules": {"pii": {"args": {"thresh": 1}}}}, r"unknown \['thresh'\]"),
    ({**MIN, "rules": {"pii": {"severity": "urgent"}}}, "severity must be one of"),
    ({**MIN, "rules": {"pii": {"colour": "red"}}}, "unknown settings"),
    ({**MIN, "mode": "enforce"}, "not supported yet"),
    ({**MIN, "profile": "paranoid"}, "profile must be one of"),
    ({**MIN, "profile": "none"}, "no rules enabled"),
    ({**MIN, "extra": 1}, "unknown keys"),
])
def test_policy_errors_are_specific(spec, match):
    with pytest.raises(PolicyError, match=match):
        PolicyPack.from_dict(spec)


def test_digest_tracks_anything_that_changes_results():
    a = PolicyPack.from_dict(MIN).digest
    assert PolicyPack.from_dict(MIN).digest == a  # stable
    assert PolicyPack.from_dict({**MIN, "rules": {"pii": {"args": {"threshold": 0.3}}}}).digest != a
    assert PolicyPack.from_dict({**MIN, "rules": {"pii": {"severity": "low"}}}).digest != a
    assert PolicyPack.from_dict({**MIN, "rules": {"pii": {"enabled": False}}}).digest != a
    assert PolicyPack.from_dict({**MIN, "version": "2"}).digest != a


def test_digest_includes_rule_code(monkeypatch):
    """Rewording a rule's questions changes its verdicts, so it must change the digest."""
    import evaltrix.governance.policy as policy

    before = PolicyPack.from_dict(MIN).digest
    fps = PolicyPack.from_dict(MIN).rule_fingerprints
    assert len(set(fps.values())) > 1  # rules in different modules fingerprint differently
    real = policy.rule_fingerprint
    monkeypatch.setattr(policy, "rule_fingerprint", lambda cls: real(cls) + ("x" if cls.name == "pii" else ""))
    assert PolicyPack.from_dict(MIN).digest != before


# --------------------------------------------------------------------------- observe


async def test_records_carry_policy_agent_and_decisions():
    pack = PolicyPack.load(EXAMPLE)
    fake = FakeJev(fixed={"tool_risk.action": "block", "delegation_scope.in_scope": 0.1})
    run = await observe(_traces("subagent_overreach"), pack, jev=fake)
    assert {r.policy_digest for r in run.records} == {pack.digest}
    assert all(r.trace_metadata == {"timestamp": "2026-10-03T10:00:00Z", "agent_version": "a@1"} for r in run.records)
    refund = next(r for r in run.records if r.rule == "tool_risk" and r.target.endswith("issue_refund"))
    assert refund.decision == "block" and refund.finding and refund.agent == "support"
    scope = next(r for r in run.records if r.rule == "delegation_scope")
    assert scope.agent == "support" and scope.finding
    undeclared = next(r for r in run.records if r.rule == "undeclared_tool" and r.finding)
    assert "OWASP-LLM06:2025 Excessive Agency" in undeclared.frameworks
    s = run.summary()
    assert s["by_agent"]["support"] >= 3 and s["traces_with_findings"] == 1
    assert run.fails("high") and not PolicyPack.from_dict(MIN).fail_on


async def test_evidence_is_redacted_before_it_is_stored():
    pack = PolicyPack.from_dict({**MIN, "profile": "none", "rules": {"pii": {}, "secrets_exposure": {}}})
    fake = FakeJev(fixed={"pii.*": 0.95, "secrets_exposure.*": 0.95})
    run = await observe(_traces("other_customer_pii", "secret_in_output"), pack, jev=fake)
    stored = json.dumps([r.model_dump(mode="json") for r in run.records]) + json.dumps(run.traces)
    assert "jane.doe@gmail.com" not in stored and "Xk9#mQ2$vL8pR4tZ" not in stored
    assert "[EMAIL]" in stored
    assert {r.rule for r in run.findings} == {"pii", "secrets_exposure"}

    raw = PolicyPack.from_dict({**MIN, "profile": "none", "rules": {"pii": {}},
                                "record": {"redact_evidence": False}})
    run = await observe(_traces("other_customer_pii"), raw, jev=FakeJev(fixed={"pii.*": 0.95}))
    assert "jane.doe@gmail.com" in json.dumps([r.model_dump(mode="json") for r in run.records])


async def test_errors_are_coverage_gaps_not_findings_or_passes():
    class Down:
        name = "down"

        async def ask(self, state, questions):
            raise ConnectionError("unreachable")

    run = await observe(_traces("clean_lookup"), PolicyPack.from_dict(MIN), jev=Down())
    assert run.errors and not run.findings
    assert all(r.status in ("error", "pass", "skipped") for r in run.records)
    assert {r.rule for r in run.errors} >= {"grounded", "tool_risk"}  # everything that needed Jev
    assert run.summary()["errors"] == len(run.errors)


# --------------------------------------------------------------------------- audit + report


async def test_audit_log_chains_runs_and_detects_tampering(tmp_path):
    pack = PolicyPack.from_dict(MIN)
    log = AuditLog(tmp_path / "audit.jsonl")
    run1 = await observe(_traces("clean_lookup"), pack, jev=FakeJev())
    run2 = await observe(_traces("retry_loop"), pack, jev=FakeJev())
    log.append(run1)
    head = log.append(run2)
    ok, n, msg = log.verify()
    assert ok and n == len(run1.records) + len(run2.records) + 2 and head[:12] in msg
    lines = log.path.read_text().splitlines()
    assert json.loads(lines[len(run1.records)])["kind"] == "run"

    row = json.loads(lines[1])
    row["status"] = "pass" if row["status"] != "pass" else "fail"
    lines[1] = json.dumps(row)
    log.path.write_text("\n".join(lines) + "\n")
    ok, _, msg = log.verify()
    assert not ok and "line 2" in msg

    lines = [json.loads(x) for x in log.path.read_text().splitlines()]
    log.path.write_text("\n".join(json.dumps(x) for x in lines[:3] + lines[4:]) + "\n")  # delete a record
    assert not log.verify()[0]


async def test_report_renders_breakdowns_and_escapes(tmp_path):
    pack = PolicyPack.load(EXAMPLE)
    traces = _traces("subagent_overreach", "clean_lookup")
    traces[1]["messages"] = [*traces[1]["messages"][:-1], {"role": "assistant", "content": "<script>alert(1)</script>"}]
    run = await observe(traces, pack, jev=FakeJev(fixed={"tool_risk.action": "block"}))
    page = render(run, pack, tmp_path / "r.html", traces=traces).read_text()
    assert "<script>alert(1)" not in page and "&lt;script&gt;" in page
    for needle in ("Where the findings are", "OWASP-LLM06:2025", ">Agent<", "would block", "agent support",
                   pack.digest[:12], "handoff"):
        assert needle in page, needle
