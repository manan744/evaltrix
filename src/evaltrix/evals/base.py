"""The Eval base class and the Result every eval produces.

An eval answers one governance or quality question about one unit (see trace/targets.py):

    class Grounded(Eval):
        name = "grounded"
        target = "trace"

        def applies(self, u):         return None if u.trace.tool_results else "no tool results"
        def precheck(self, u):        ...   # optional: decide in code, no Jev call
        def state(self, u):           return {"evidence": ..., "claims": [...]}
        def questions(self, u):       return {"c0": YesNo("Is claims[0] supported by evidence?")}
        def reduce(self, answers, u): return Result(score=..., passed=...)

The engine fills in `eval`, `target`, `answers`, `usage`, and turns exceptions into error results.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from ..jev.types import Answer, Question, Usage
from ..trace.targets import Target, Unit

Status = Literal["pass", "fail", "skipped", "error"]
Category = Literal["security", "privacy", "action_safety", "honesty", "reliability", "custom"]
Severity = Literal["critical", "high", "medium", "low"]


class Result(BaseModel):
    eval: str = ""
    target: str = ""  # unit key: "trace", "tool_call[0]:lookup_order", ...
    agent: str | None = None  # the agent the unit belongs to (multi-agent traces)
    score: float | None = None  # 0..1, higher is better
    passed: bool | None = None  # None when the eval only measures
    label: str | None = None  # for choice-shaped evals: the option Jev picked
    p: float | None = None  # the headline probability, when there is one
    detail: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    answers: dict[str, Answer] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)  # this eval's share of the request it was packed into
    model: str | None = None  # Jev version that answered
    skipped: bool = False
    error: str | None = None

    @property
    def status(self) -> Status:
        if self.error is not None:
            return "error"
        if self.skipped:
            return "skipped"
        return "fail" if self.passed is False else "pass"

    def __str__(self) -> str:
        bits = [f"{self.eval}@{self.target}", self.status]
        if self.label is not None:
            bits.append(self.label)
        if self.score is not None:
            bits.append(f"score={self.score:.2f}")
        if self.p is not None:
            bits.append(f"p={self.p:.2f}")
        if self.error or self.detail:
            bits.append(self.error or self.detail)
        return "  ".join(bits)


class Eval:
    """Subclass and set `name` and `target`; implement `questions` and `reduce` (and usually `state`)."""

    name: ClassVar[str] = ""
    target: ClassVar[Target] = "trace"
    category: ClassVar[Category] = "custom"
    severity: ClassVar[Severity] = "medium"  # default; a governance policy can override it per deployment
    frameworks: ClassVar[tuple[str, ...]] = ()  # standards this rule produces evidence for (see evals/frameworks.py)

    def __init_subclass__(cls, **kw: Any) -> None:
        super().__init_subclass__(**kw)
        if not cls.name:
            raise TypeError(f"{cls.__name__} must set a `name`")
        if "." in cls.name:
            raise TypeError(f"{cls.__name__}.name must not contain '.' (it namespaces question ids)")

    def applies(self, u: Unit) -> str | None:
        """None if this eval can judge the unit, else the reason it can't (recorded as skipped)."""
        return None

    def precheck(self, u: Unit) -> Result | None:
        """Decide without Jev when code can (no PII found, no claims to check). None to ask Jev."""
        return None

    def state(self, u: Unit) -> dict[str, Any]:
        """What Jev reads. Keys are shared with other evals in the same request; equal values merge."""
        return {}

    def questions(self, u: Unit) -> dict[str, Question]:
        raise NotImplementedError

    def reduce(self, answers: dict[str, Answer], u: Unit) -> Result:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"
