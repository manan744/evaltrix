"""Run a policy pack over traces and turn every result into an audit record.

    run = await observe(traces, pack, jev=Jev())
    run.findings, run.summary(), run.records
    AuditLog("audit/evaluations.jsonl").append(run)

Observe mode only: nothing is blocked. A record whose rule decided "escalate" or "block" (tool_risk)
carries that decision, so you can see what enforcement would have done before switching it on.

A record is a *finding* when its rule failed. Errors are not findings, but they are not passes either:
they are coverage gaps, counted and reported separately.
"""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable

from pydantic import BaseModel, Field

from .. import __version__
from ..detect import find_pii, find_secrets, mask_secrets, redact
from ..engine.run import TraceReport, evaluate_many
from ..evals.base import Result
from ..jev.client import Backend
from ..jev.types import Usage
from ..trace.model import Trace
from ..trace.normalize import to_trace
from .policy import SEVERITIES, PolicyPack

SEVERITY_RANK = {s: len(SEVERITIES) - i for i, s in enumerate(SEVERITIES)}  # critical=4 ... low=1


class Record(BaseModel):
    """One rule's verdict on one unit of one trace. The unit of the audit trail."""

    kind: str = "evaluation"
    record_id: str
    run_id: str
    trace_id: str | None
    trace_metadata: dict[str, Any] = Field(default_factory=dict)
    rule: str
    category: str
    severity: str
    owner: str
    frameworks: list[str]
    target: str
    agent: str | None
    status: str  # pass | fail | skipped | error
    finding: bool
    decision: str | None  # tool_risk: allow | escalate | block
    score: float | None
    p: float | None
    detail: str
    evidence: dict[str, Any]
    model: str | None
    usage: Usage
    policy_pack: str
    policy_version: str
    policy_digest: str
    evaltrix_version: str = __version__
    evaluated_at: str


def _redact(value: Any) -> Any:
    """Strip personal data and secrets from evidence before it is stored."""
    if isinstance(value, str):
        secrets = find_secrets(value)
        if secrets:
            value = mask_secrets(value, secrets)
        pii = find_pii(value)
        return redact(value, pii) if pii else value
    if isinstance(value, dict):
        return {k: _redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(v) for v in value]
    return value


class Run(BaseModel):
    run_id: str
    started_at: str
    pack: str
    policy_version: str
    policy_digest: str
    traces: list[dict[str, Any]]  # id + the recorded metadata, per trace observed
    records: list[Record]
    usage: Usage = Field(default_factory=Usage)
    models: list[str] = Field(default_factory=list)

    @property
    def findings(self) -> list[Record]:
        return [r for r in self.records if r.finding]

    @property
    def errors(self) -> list[Record]:
        return [r for r in self.records if r.status == "error"]

    def fails(self, at_or_above: str) -> bool:
        return any(SEVERITY_RANK[r.severity] >= SEVERITY_RANK[at_or_above] for r in self.findings)

    def summary(self) -> dict[str, Any]:
        by_rule: dict[str, dict[str, Any]] = {}
        for r in self.records:
            row = by_rule.setdefault(r.rule, {"rule": r.rule, "category": r.category, "severity": r.severity,
                                              "owner": r.owner, "frameworks": r.frameworks, "evaluated": 0,
                                              "findings": 0, "skipped": 0, "errors": 0, "decisions": Counter()})
            if r.status in ("pass", "fail"):
                row["evaluated"] += 1
            row["findings"] += r.finding
            row["skipped"] += r.status == "skipped"
            row["errors"] += r.status == "error"
            if r.decision:
                row["decisions"][r.decision] += 1
        for row in by_rule.values():
            row["pass_rate"] = 1 - row["findings"] / row["evaluated"] if row["evaluated"] else None
            row["decisions"] = dict(row["decisions"])

        def count(key) -> dict[str, int]:
            c: Counter[str] = Counter()
            for r in self.findings:
                for k in key(r):
                    c[k] += 1
            return dict(c.most_common())

        traces_with = {r.trace_id for r in self.findings}
        return {
            "traces": len(self.traces),
            "traces_with_findings": len(traces_with),
            "findings": len(self.findings),
            "errors": len(self.errors),
            "by_severity": {s: sum(r.severity == s for r in self.findings) for s in SEVERITIES},
            "by_category": count(lambda r: [r.category]),
            "by_framework": count(lambda r: r.frameworks),
            "by_agent": count(lambda r: [r.agent] if r.agent else []),
            "by_rule": list(by_rule.values()),
            "usage": self.usage.model_dump(),
            "models": self.models,
        }


def _record(run_id: str, t: Trace, r: Result, pack: PolicyPack, now: str) -> Record:
    cfg = pack.config(r.eval)
    evidence = _redact(r.evidence) if pack.redact_evidence else r.evidence
    detail = _redact(r.error or r.detail) if pack.redact_evidence else (r.error or r.detail)
    return Record(
        record_id=uuid.uuid4().hex, run_id=run_id, trace_id=t.id,
        trace_metadata={k: t.metadata[k] for k in pack.record_metadata if k in t.metadata},
        rule=r.eval, category=cfg.rule.category, severity=cfg.severity, owner=cfg.owner,
        frameworks=list(cfg.frameworks), target=r.target, agent=r.agent, status=r.status,
        finding=r.status == "fail", decision=r.label if r.eval == "tool_risk" else None,
        score=r.score, p=r.p, detail=detail, evidence=evidence, model=r.model, usage=r.usage,
        policy_pack=pack.name, policy_version=pack.version, policy_digest=pack.digest, evaluated_at=now,
    )


async def observe(traces: Iterable[Any], pack: PolicyPack, jev: Backend, concurrency: int = 8) -> Run:
    normalized = [to_trace(t) for t in traces]
    run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    started = datetime.now(timezone.utc).isoformat()
    reports: list[TraceReport] = await evaluate_many(normalized, pack.rules, jev=jev, concurrency=concurrency)
    records: list[Record] = []
    usage = Usage()
    models: list[str] = []
    for t, rep in zip(normalized, reports):
        now = datetime.now(timezone.utc).isoformat()
        records += [_record(run_id, t, r, pack, now) for r in rep.results]
        usage = usage + rep.usage
        models += [m for m in rep.models if m not in models]
    return Run(
        run_id=run_id, started_at=started, pack=pack.name, policy_version=pack.version, policy_digest=pack.digest,
        traces=[{"trace_id": t.id, **{k: t.metadata[k] for k in pack.record_metadata if k in t.metadata},
                 "agents": t.agent_names, "first_user_message": _redact(t.first_user_message)[:200]}
                for t in normalized],
        records=records, usage=usage, models=models,
    )


def group_by_trace(records: list[Record]) -> dict[str | None, list[Record]]:
    out: dict[str | None, list[Record]] = defaultdict(list)
    for r in records:
        out[r.trace_id].append(r)
    return out
