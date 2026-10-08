"""A Jev stand-in for tests and CI: no network, deterministic, rule-driven.

    FakeJev()                                         # every yes/no 0.5, every choice/score uniform
    FakeJev(fixed={"grounded.c1": 0.05})              # exact question ids ("*" globs allowed)
    FakeJev(rules=lambda qid, q, state: ...)          # your judgement; return None to fall through

A rule or fixed value can be: a float (yes/no), an option name or {option: p} (choice),
a level index or [p0, p1, ...] (score). Every call is kept in `.calls` for assertions.
"""

from __future__ import annotations

import fnmatch
import json
from collections.abc import Callable, Mapping
from typing import Any

from .types import Answer, Choice, ChoiceAnswer, JevResponse, Question, Score, ScoreAnswer, Usage, YesNo, YesNoAnswer

Rule = Callable[[str, Question, Any], Any]


class FakeJev:
    name = "fake"

    def __init__(self, rules: Rule | None = None, fixed: Mapping[str, Any] | None = None, default_yes: float = 0.5):
        self.rules = rules
        self.fixed = dict(fixed or {})
        self.default_yes = default_yes
        self.calls: list[tuple[Any, dict[str, Question]]] = []

    def _value(self, qid: str, q: Question, state: Any) -> Any:
        if qid in self.fixed:
            return self.fixed[qid]
        for pattern, v in self.fixed.items():
            if fnmatch.fnmatchcase(qid, pattern):
                return v
        return self.rules(qid, q, state) if self.rules else None

    def _answer(self, v: Any, q: Question, qid: str) -> Answer:
        if isinstance(q, YesNo):
            if isinstance(v, bool):
                v = 0.95 if v else 0.05
            return YesNoAnswer(p=self.default_yes if v is None else float(v))
        if isinstance(q, Choice):
            opts = list(q.options)
            if v is None:
                return ChoiceAnswer(probs={o: 1.0 for o in opts})
            if isinstance(v, str):
                if v not in q.options:
                    raise ValueError(f"FakeJev: {qid} has no option {v!r}")
                rest = 0.1 / (len(opts) - 1)
                return ChoiceAnswer(probs={o: 0.9 if o == v else rest for o in opts})
            return ChoiceAnswer(probs={o: float(v.get(o, 0.0)) for o in opts})
        if isinstance(q, Score):
            n = len(q.levels)
            if v is None:
                return ScoreAnswer(probs=[1.0] * n)
            if isinstance(v, int) and not isinstance(v, bool):
                if not 0 <= v < n:
                    raise ValueError(f"FakeJev: {qid} level {v} out of range 0..{n - 1}")
                return ScoreAnswer(probs=[1.0 if i == v else 0.0 for i in range(n)])
            return ScoreAnswer(probs=list(v))
        raise TypeError(q)

    async def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        self.calls.append((state, dict(questions)))
        answers = {qid: self._answer(self._value(qid, q, state), q, qid) for qid, q in questions.items()}
        approx_tokens = len(json.dumps({"state": state, "q": {k: q.model_dump() for k, q in questions.items()}},
                                       default=str)) // 4
        return JevResponse(answers=answers, usage=Usage(requests=1, input_tokens=approx_tokens))
