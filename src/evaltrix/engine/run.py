"""evaluate(): judge one trace with a set of evals. evaluate_many(): a dataset, concurrently.

    report = await evaluate(trace, [Grounded(), IndirectInjection(), ToolRisk()], jev=Jev())
    report["grounded"]                       # the trace-level result
    report.for_eval("tool_risk")             # one result per tool call
    report.failures, report.errors, report.usage

Each trace is split into units (trace / tool_call / tool_result), each unit's evals are packed
(engine/pack.py), and all of a trace's requests go out concurrently. Failures never turn into
passes: a failed request, a malformed answer or an exception in an eval becomes an `error` result.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import BaseModel, Field

from ..evals.base import Eval, Result
from ..jev.client import Backend, run_sync
from ..jev.types import Usage
from ..trace.model import Trace
from ..trace.normalize import to_trace
from ..trace.targets import units
from .pack import Request, plan


class TraceReport(BaseModel):
    trace_id: str | None = None
    results: list[Result]
    usage: Usage = Field(default_factory=Usage)  # totals across this trace's requests
    models: list[str] = Field(default_factory=list)  # Jev versions that answered

    def __getitem__(self, eval_name: str) -> Result:
        """The trace-level result of an eval. Use for_eval() for tool_call / tool_result evals."""
        for r in self.results:
            if r.eval == eval_name and r.target == "trace":
                return r
        raise KeyError(f"no trace-level result for {eval_name!r}")

    def for_eval(self, eval_name: str) -> list[Result]:
        return [r for r in self.results if r.eval == eval_name]

    @property
    def failures(self) -> list[Result]:
        return [r for r in self.results if r.status == "fail"]

    @property
    def errors(self) -> list[Result]:
        return [r for r in self.results if r.status == "error"]

    @property
    def passed(self) -> bool:
        """True only if nothing failed and nothing errored."""
        return not self.failures and not self.errors

    def table(self) -> str:
        w = max((len(f"{r.eval}@{r.target}") for r in self.results), default=10) + 2
        lines = []
        for r in self.results:
            head = r.label or (f"{r.score:.2f}" if r.score is not None else "")
            p = f"p={r.p:.2f}" if r.p is not None else ""
            lines.append(f"{r.eval + '@' + r.target:<{w}} {r.status:<8} {head:<14} {p:<7} {r.error or r.detail}".rstrip())
        lines += ["", str(self.usage)]
        return "\n".join(lines)


def _split(total: int, weights: list[int]) -> list[int]:
    """Split an integer total by weight so the parts add up exactly (largest remainder)."""
    s = sum(weights)
    raw = [total * w / s for w in weights]
    parts = [int(x) for x in raw]
    for i in sorted(range(len(raw)), key=lambda i: raw[i] - parts[i], reverse=True)[: total - sum(parts)]:
        parts[i] += 1
    return parts


async def _send(req: Request, jev: Backend) -> tuple[list[Result], Usage, str | None]:
    u = req.unit
    # canonical order: the same unit and rules always produce the byte-identical request, whatever order a
    # policy lists its rules in. Borderline answers shift slightly with request context, so this is what
    # makes a re-run under the same policy digest reproduce the same verdicts.
    state = dict(sorted(req.state.items()))
    questions = dict(sorted(req.questions.items()))
    try:
        resp = await jev.ask(state, questions)
    except Exception as e:  # noqa: BLE001 - recorded on every eval in the request
        msg = f"jev request failed: {type(e).__name__}: {e}"
        return [Result(eval=m.eval.name, target=u.key, agent=u.agent, error=msg) for m in req.members], Usage(), None

    weights = [m.weight for m in req.members]
    tokens = _split(resp.usage.input_tokens, weights)
    total_w = sum(weights)
    results = []
    for m, tok in zip(req.members, tokens):
        share = Usage(input_tokens=tok, cost_usd=resp.usage.cost_usd * m.weight / total_w,
                      latency_ms=resp.usage.latency_ms)  # every member waited for the whole request
        missing = [qid for qid in m.questions if f"{m.eval.name}.{qid}" not in resp.answers]
        answers = {qid: resp.answers[f"{m.eval.name}.{qid}"] for qid in m.questions if qid not in missing}
        if missing:
            r = Result(eval=m.eval.name, target=u.key, agent=u.agent, error=f"jev returned no answer for {missing}")
        else:
            try:
                r = m.eval.reduce(answers, u)
                r = r.model_copy(update={"eval": m.eval.name, "target": u.key, "agent": u.agent})
            except Exception as e:  # noqa: BLE001
                r = Result(eval=m.eval.name, target=u.key, agent=u.agent, error=f"reduce: {type(e).__name__}: {e}")
        r.answers, r.usage, r.model = answers, share, resp.model
        results.append(r)
    return results, resp.usage, resp.model


def _check_names(evals: Sequence[Eval]) -> None:
    seen: dict[tuple[str, str], Eval] = {}
    for ev in evals:
        key = (ev.name, ev.target)
        if key in seen:
            raise ValueError(f"two evals named {ev.name!r} on target {ev.target!r}: {seen[key]!r}, {ev!r}")
        seen[key] = ev


async def evaluate(trace: Any, evals: Sequence[Eval], jev: Backend) -> TraceReport:
    _check_names(evals)
    t: Trace = to_trace(trace)
    kinds = {ev.target for ev in evals}
    requests: list[Request] = []
    results: list[Result] = []
    for unit in units(t, kinds):
        reqs, decided = plan(unit, list(evals))
        requests += reqs
        results += decided
    usage = Usage()
    models: list[str] = []
    for rs, u, model in await asyncio.gather(*(_send(r, jev) for r in requests)):
        results += rs
        usage = usage + u
        if model and model not in models:
            models.append(model)
    # stable order: by eval as given, then by unit
    order = {ev.name: i for i, ev in enumerate(evals)}
    results.sort(key=lambda r: (order.get(r.eval, len(order)), r.target != "trace", r.target))
    return TraceReport(trace_id=t.id, results=results, usage=usage, models=models)


async def evaluate_many(traces: Iterable[Any], evals: Sequence[Eval], jev: Backend,
                        concurrency: int = 8) -> list[TraceReport]:
    """One report per trace, in input order. At most `concurrency` traces in flight at once."""
    _check_names(evals)
    sem = asyncio.Semaphore(concurrency)

    async def one(t: Any) -> TraceReport:
        async with sem:
            return await evaluate(t, evals, jev)

    return list(await asyncio.gather(*(one(t) for t in traces)))


def evaluate_sync(trace: Any, evals: Sequence[Eval], jev: Backend) -> TraceReport:
    return run_sync(evaluate(trace, evals, jev))
