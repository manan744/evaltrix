"""Typed questions Jev can answer, and the answers it returns.

Three kinds, nothing else:

    YesNo("Is claims[0] supported by evidence?")                        -> YesNoAnswer.p
    Choice("Should this call run?", options={"approve": ..., ...})      -> ChoiceAnswer.probs
    Score("How relevant is answer to question?", levels=[low, ..., high]) -> ScoreAnswer.probs

Instructions refer to keys of the state by name; Jev reads the state and judges. Describe each
option or level in words ("Irreversible or moves money") rather than just labelling it ("risky"):
the description is what Jev matches the state against.
"""

from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, Field, field_validator, model_validator

Content = Union[str, dict, list]  # Jev accepts structured instructions/criteria as well as text

MAX_CHOICE_OPTIONS = 255
MAX_SCORE_LEVELS = 10


# --------------------------------------------------------------------------- questions


class YesNo(BaseModel):
    kind: Literal["yes_no"] = "yes_no"
    instructions: Content
    true: Content | None = None  # what "yes" means, optional but sharpens the answer
    false: Content | None = None

    def __init__(self, instructions: Content | None = None, /, **kw: Any):
        if instructions is not None:
            kw["instructions"] = instructions
        super().__init__(**kw)

    @model_validator(mode="after")
    def _both_or_neither(self) -> YesNo:
        if (self.true is None) != (self.false is None):
            raise ValueError("give both `true` and `false` descriptions, or neither")
        return self


class Choice(BaseModel):
    kind: Literal["choice"] = "choice"
    instructions: Content
    options: dict[str, Content]

    def __init__(self, instructions: Content | None = None, /, **kw: Any):
        if instructions is not None:
            kw["instructions"] = instructions
        super().__init__(**kw)

    @field_validator("options")
    @classmethod
    def _n_options(cls, v: dict[str, Content]) -> dict[str, Content]:
        if not 2 <= len(v) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice needs 2..{MAX_CHOICE_OPTIONS} options, got {len(v)}")
        return v


class Score(BaseModel):
    kind: Literal["score"] = "score"
    instructions: Content
    levels: list[Content]  # ordered, lowest first

    def __init__(self, instructions: Content | None = None, /, **kw: Any):
        if instructions is not None:
            kw["instructions"] = instructions
        super().__init__(**kw)

    @field_validator("levels")
    @classmethod
    def _n_levels(cls, v: list[Content]) -> list[Content]:
        if not 2 <= len(v) <= MAX_SCORE_LEVELS:
            raise ValueError(f"score needs 2..{MAX_SCORE_LEVELS} levels, got {len(v)}")
        return v


Question = Union[YesNo, Choice, Score]


# --------------------------------------------------------------------------- answers


def _normalized(values: list[float]) -> list[float]:
    vals = [min(1.0, max(0.0, float(v))) for v in values]
    total = sum(vals)
    if total <= 0:
        return [1.0 / len(vals)] * len(vals)
    return [v / total for v in vals]


def _confidence(probs: list[float]) -> float:
    """How far the top probability is above uniform, 0 (no idea) .. 1 (certain)."""
    k = len(probs)
    return 1.0 if k < 2 else (max(probs) - 1 / k) / (1 - 1 / k)


class YesNoAnswer(BaseModel):
    kind: Literal["yes_no"] = "yes_no"
    p: float = Field(ge=0.0, le=1.0)  # P(yes)

    @property
    def yes(self) -> bool:
        return self.p >= 0.5


class ChoiceAnswer(BaseModel):
    kind: Literal["choice"] = "choice"
    probs: dict[str, float]
    jev_confidence: float | None = None  # Jev's own confidence, when it sends one

    @model_validator(mode="after")
    def _norm(self) -> ChoiceAnswer:
        self.probs = dict(zip(self.probs, _normalized(list(self.probs.values()))))
        return self

    @property
    def top(self) -> str:
        return max(self.probs, key=self.probs.__getitem__)

    def p(self, option: str) -> float:
        return self.probs.get(option, 0.0)

    @property
    def confidence(self) -> float:
        if self.jev_confidence is not None:
            return self.jev_confidence
        return _confidence(list(self.probs.values()))


class ScoreAnswer(BaseModel):
    kind: Literal["score"] = "score"
    probs: list[float]  # probs[i] = P(level i)
    jev_confidence: float | None = None

    @model_validator(mode="after")
    def _norm(self) -> ScoreAnswer:
        self.probs = _normalized(self.probs)
        return self

    @property
    def expected(self) -> float:
        """Expected level index, 0 .. n-1, fractional."""
        return sum(i * p for i, p in enumerate(self.probs))

    @property
    def normalized(self) -> float:
        """Expected level rescaled to 0..1."""
        return self.expected / (len(self.probs) - 1)

    @property
    def top(self) -> int:
        return max(range(len(self.probs)), key=self.probs.__getitem__)

    @property
    def confidence(self) -> float:
        if self.jev_confidence is not None:
            return self.jev_confidence
        return _confidence(self.probs)


Answer = Union[YesNoAnswer, ChoiceAnswer, ScoreAnswer]


# --------------------------------------------------------------------------- usage / response


class Usage(BaseModel):
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            requests=self.requests + other.requests,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            latency_ms=self.latency_ms + other.latency_ms,
            cost_usd=self.cost_usd + other.cost_usd,
        )

    def __str__(self) -> str:
        return (f"{self.requests} request{'s' * (self.requests != 1)} · {self.input_tokens:,} tokens · "
                f"${self.cost_usd:.6f} · {self.latency_ms / 1000:.2f}s")


class JevResponse(BaseModel):
    answers: dict[str, Answer]
    model: str | None = None  # the version that actually answered, e.g. "jev-1.13.0"; keep it for audits
    usage: Usage = Field(default_factory=Usage)
    raw: Any = None
