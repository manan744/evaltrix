"""Pack the evals for one unit into as few Jev requests as possible.

Every eval contributes a state dict and some questions. Evals whose states agree on every shared key
go in the same request: states are merged, question ids are namespaced "<eval>.<qid>". If an eval
needs a different value for a key another eval already set, it starts a new request rather than
overwriting it, so no eval ever sees a state it didn't ask for under a key it reads.

One unit never shares a request with another unit, and one trace never shares with another trace.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..evals.base import Eval, Result
from ..jev.types import Question
from ..trace.targets import Unit


@dataclass
class Member:
    eval: Eval
    questions: dict[str, Question]  # un-namespaced, as the eval wrote them
    weight: int  # rough size of what this eval added, for splitting the request's token cost


@dataclass
class Request:
    unit: Unit
    state: dict[str, Any] = field(default_factory=dict)
    questions: dict[str, Question] = field(default_factory=dict)  # namespaced
    members: list[Member] = field(default_factory=list)

    def accepts(self, state: dict[str, Any]) -> bool:
        return all(k not in self.state or self.state[k] == v for k, v in state.items())

    def add(self, ev: Eval, state: dict[str, Any], qs: dict[str, Question]) -> None:
        self.state.update(state)
        for qid, q in qs.items():
            self.questions[f"{ev.name}.{qid}"] = q
        size = len(json.dumps(state, sort_keys=True, default=str)) + sum(
            len(json.dumps(q.model_dump(), default=str)) for q in qs.values())
        self.members.append(Member(ev, qs, max(1, size)))


def _err(ev: Eval, u: Unit, where: str, e: BaseException) -> Result:
    return Result(eval=ev.name, target=u.key, agent=u.agent, error=f"{where}: {type(e).__name__}: {e}")


def plan(unit: Unit, evals: list[Eval]) -> tuple[list[Request], list[Result]]:
    """Requests to send for this unit, plus results already decided without Jev (skipped, prechecked, errors)."""
    requests: list[Request] = []
    decided: list[Result] = []
    for ev in evals:
        if ev.target != unit.kind:
            continue
        try:
            reason = ev.applies(unit)
            if reason is not None:
                decided.append(Result(eval=ev.name, target=unit.key, agent=unit.agent, skipped=True, detail=reason))
                continue
            pre = ev.precheck(unit)
        except Exception as e:  # noqa: BLE001 - one broken eval must not take down the others
            decided.append(_err(ev, unit, "precheck", e))
            continue
        if pre is not None:
            decided.append(pre.model_copy(update={"eval": ev.name, "target": unit.key, "agent": unit.agent}))
            continue
        try:
            state = ev.state(unit)
            qs = ev.questions(unit)
        except Exception as e:  # noqa: BLE001
            decided.append(_err(ev, unit, "state/questions", e))
            continue
        if not qs:
            decided.append(Result(eval=ev.name, target=unit.key, agent=unit.agent, error="questions() returned nothing; "
                                  "use precheck() to decide without Jev"))
            continue
        req = next((r for r in requests if r.accepts(state)), None)
        if req is None:
            req = Request(unit)
            requests.append(req)
        req.add(ev, state, qs)
    return requests, decided
