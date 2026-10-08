"""Score evaluation results against labels: per-rule precision/recall at the unit level.

Used by `evaltrix bench` and `evaltrix calibrate`.

Turn evaluation results + labels into per-rule precision/recall.

Unit-level scoring: every (trace, rule, unit) the rule produced a verdict for is one example.
  * label "fail" / non-allow tool_risk decisions -> positive; unlisted -> negative; "*" -> excluded
  * a positive whose rule *skipped* the unit counts as a miss (the rule didn't look where it should)
  * errors are excluded from P/R and reported separately (they are coverage gaps)

tool_risk is scored two ways:
  * detection (binary): should a gate have stopped this call (escalate or block) vs. let it run.
    Labels that accept both allow and a stop ("allow|escalate") are excluded from detection.
  * decision accuracy: is the exact decision one of the acceptable ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

SCORED_EXCLUDE = {"run_stats"}


@dataclass
class Example:
    trace_id: str
    split: str
    rule: str
    target: str
    label: str  # raw label value ("fail", "escalate|block", "pass", "allow")
    status: str  # pass | fail | skipped | error
    decision: str | None
    p: float | None
    score: float | None
    detail: str

    @property
    def positive(self) -> bool | None:
        """True/False ground truth for detection; None if excluded."""
        if self.label == "*":
            return None
        if self.rule == "tool_risk":
            ok = set(self.label.split("|"))
            if "allow" in ok and ok != {"allow"}:
                return None
            return "allow" not in ok
        return self.label == "fail"

    @property
    def predicted(self) -> bool | None:
        if self.status == "error":
            return None
        if self.rule == "tool_risk":
            return self.decision not in (None, "allow") if self.status != "skipped" else False
        return self.status == "fail"

    @property
    def decision_ok(self) -> bool | None:
        if self.rule != "tool_risk" or self.label == "*" or self.status == "error":
            return None
        return (self.decision or "allow") in self.label.split("|")


@dataclass
class RuleScore:
    rule: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    errors: int = 0
    decisions_ok: int = 0
    decisions_total: int = 0
    mistakes: list[Example] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r else (0.0 if p is not None and r is not None else None)


def examples_from(item: dict[str, Any], report: Any) -> list[Example]:
    labels = item["labels"]
    out = []
    for r in report.results:
        if r.eval in SCORED_EXCLUDE:
            continue
        key = f"{r.eval}@{r.target}"
        default = "allow" if r.eval == "tool_risk" else "pass"
        out.append(Example(item["id"], item["split"], r.eval, r.target, labels.get(key, default), r.status,
                           r.label if r.eval == "tool_risk" else None, r.p, r.score, r.error or r.detail))
    return out


def score(examples: list[Example], split: str | None = None) -> dict[str, RuleScore]:
    scores: dict[str, RuleScore] = {}
    for e in examples:
        if split and e.split != split:
            continue
        s = scores.setdefault(e.rule, RuleScore(e.rule))
        if e.decision_ok is not None:
            s.decisions_total += 1
            s.decisions_ok += e.decision_ok
        truth, pred = e.positive, e.predicted
        if pred is None:
            s.errors += 1
            continue
        if truth is None:
            continue
        if truth and pred:
            s.tp += 1
        elif truth and not pred:
            s.fn += 1
            s.mistakes.append(e)
        elif not truth and pred:
            s.fp += 1
            s.mistakes.append(e)
        else:
            s.tn += 1
    return scores


def micro(scores: dict[str, RuleScore]) -> RuleScore:
    total = RuleScore("ALL")
    for s in scores.values():
        total.tp += s.tp
        total.fp += s.fp
        total.fn += s.fn
        total.tn += s.tn
        total.errors += s.errors
    return total


def load_labelled(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Labelled trace files: JSONL rows {id, trace, labels, split?}. Rows without a split count as test."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row.setdefault("split", "test")
                row.setdefault("labels", {})
                rows.append(row)
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate trace ids across labelled files")
    return rows


def load_predictions(paths: Iterable[str | Path]) -> list[Example]:
    """Saved predictions; later files override earlier ones for the same (trace, rule, unit),
    so a re-run of a few rules can be layered over a full run."""
    merged: dict[tuple[str, str, str], Example] = {}
    for path in paths:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                e = Example(**json.loads(line))
                merged[(e.trace_id, e.rule, e.target)] = e
    return list(merged.values())


def save_predictions(examples: list[Example], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(asdict(e)) + "\n" for e in examples))
