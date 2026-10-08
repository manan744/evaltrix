"""Fit per-rule thresholds on labelled predictions, and measure how trustworthy the probabilities are.

    report = calibrate(examples)               # examples from evaltrix.labelled (saved by `evaltrix bench`)
    print(report.table())
    report.policy_overrides()                  # {"rules": {name: {"args": {...}}}} for the tuned rules

Method, per rule:
  * every unit gets a risk score in [0, 1] (the rule's probability that something is wrong)
  * thresholds are fitted on the **dev** split only, over a coarse grid, and reported on **test**
  * a rule is only tuned when dev has at least `min_positives` positives and `min_negatives` negatives,
    and the fitted threshold beats the default on dev by `min_gain`; otherwise the default stays.
    Small label sets make fitted thresholds memorise noise; "keep default" is the honest answer then.
  * calibration quality: Brier score (lower is better), expected calibration error over 10 bins (lower is
    better), AUROC (how well risk ranks real problems above non-problems; 0.5 = chance, 1.0 = perfect).

Rules whose verdict is not "risk >= threshold" (grounded and action_claims pick the most likely option per
sentence; tool_risk has several knobs; undeclared_tool and run_stats are code) are measured, not tuned.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .evals.builtin import ALL_RULES
from .labelled import Example

_RULES = {c.name: c for c in ALL_RULES}

# rule -> (risk from the saved p, rule argument from a risk threshold tau, default tau)
_DIRECT = ("prompt_injection", "indirect_injection", "goal_hijacking", "system_prompt_leakage", "secrets_exposure",
           "harmful_output", "pii", "phi", "data_exfiltration", "oversight_bypass", "tool_error_masking",
           "loop_detection")
_INVERSE = ("excessive_agency", "delegation_scope", "goal_completion")  # their `threshold` is on the "ok" side


def _risk(e: Example) -> float | None:
    """The unit's probability that something is wrong, whatever the rule's own `p` means."""
    if e.rule == "tool_risk":  # p is the probability of the chosen decision; score is p(approve)
        return None if e.score is None else 1 - e.score
    if e.p is None:  # code-only verdicts: certain either way
        return {"fail": 1.0, "pass": 0.0}.get(e.status)
    return 1 - e.p if e.rule == "goal_completion" else e.p


def _param(rule: str, tau: float) -> float:
    return round(1 - tau, 4) if rule in _INVERSE else round(tau, 4)


def _default_tau(rule: str) -> float:
    t = getattr(_RULES[rule](), "threshold")
    return round(1 - t, 4) if rule in _INVERSE else t


TUNABLE = set(_DIRECT) | set(_INVERSE)
GRID = [round(0.05 * i, 2) for i in range(1, 20)]


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def f1(self) -> float:
        p, r = self.precision or 0.0, self.recall or 0.0
        return 2 * p * r / (p + r) if p + r else 0.0


def _labelled(examples: list[Example], rule: str, split: str | None) -> list[tuple[float | None, bool, str]]:
    out = []
    for e in examples:
        if e.rule != rule or (split and e.split != split) or e.status == "error" or e.positive is None:
            continue
        risk = None if e.status == "skipped" else _risk(e)
        out.append((risk, bool(e.positive), e.status))
    return out


def _counts_at(rows: list[tuple[float | None, bool, str]], tau: float | None) -> Counts:
    """tau=None: use the verdict the rule actually gave (status), i.e. the default threshold."""
    c = Counts()
    for risk, truth, status in rows:
        pred = (status == "fail") if tau is None else (risk is not None and risk >= tau)
        if truth and pred:
            c.tp += 1
        elif truth:
            c.fn += 1
        elif pred:
            c.fp += 1
        else:
            c.tn += 1
    return c


def brier(rows: list[tuple[float | None, bool, str]]) -> float | None:
    pts = [(r, t) for r, t, _ in rows if r is not None]
    return sum((r - t) ** 2 for r, t in pts) / len(pts) if pts else None


def ece(rows: list[tuple[float | None, bool, str]], bins: int = 10) -> float | None:
    pts = [(r, t) for r, t, _ in rows if r is not None]
    if not pts:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inb = [(r, t) for r, t in pts if lo <= r < hi or (b == bins - 1 and r == 1.0)]
        if inb:
            conf = sum(r for r, _ in inb) / len(inb)
            acc = sum(t for _, t in inb) / len(inb)
            total += len(inb) / len(pts) * abs(conf - acc)
    return total


def auroc(rows: list[tuple[float | None, bool, str]]) -> float | None:
    pos = [r if r is not None else 0.0 for r, t, _ in rows if t]
    neg = [r if r is not None else 0.0 for r, t, _ in rows if not t]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


Objective = Callable[[Counts], float]


def objective(name: str) -> Objective:
    """f1 | recall@precision>=X | precision@recall>=X"""
    if name == "f1":
        return lambda c: c.f1
    kind, _, floor = name.partition(">=")
    if kind == "recall@precision" and floor:
        x = float(floor)
        return lambda c: (c.recall or 0.0) if (c.precision or 0.0) >= x else -1.0
    if kind == "precision@recall" and floor:
        x = float(floor)
        return lambda c: (c.precision or 0.0) if (c.recall or 0.0) >= x else -1.0
    raise ValueError("objective must be f1, recall@precision>=X or precision@recall>=X")


@dataclass
class RuleCalibration:
    rule: str
    status: str  # tuned | keep default | insufficient data | not tunable | no data
    reason: str
    default_tau: float | None
    fitted_tau: float | None
    dev_pos: int
    dev_neg: int
    dev_default: Counts
    dev_fitted: Counts | None
    test_default: Counts
    test_fitted: Counts | None
    brier: float | None
    ece: float | None
    auroc: float | None
    override: dict[str, Any] | None = None


@dataclass
class CalibrationReport:
    rules: list[RuleCalibration] = field(default_factory=list)
    objective: str = "f1"

    def policy_overrides(self) -> dict[str, Any]:
        return {"rules": {r.rule: {"args": r.override} for r in self.rules if r.override}}

    def table(self) -> str:
        def f(x: float | None) -> str:
            return "  -  " if x is None else f"{x:5.2f}"

        lines = [f"objective: {self.objective} (fitted on dev, reported on test)",
                 f"{'rule':<22} {'status':<18} {'tau':>11}  {'dev F1':>11}  {'test P/R/F1 default':>19}  "
                 f"{'test P/R/F1 fitted':>19}  {'Brier':>5} {'ECE':>5} {'AUROC':>5}  dev +/-"]
        for r in self.rules:
            tau = f"{f(r.default_tau)}->{f(r.fitted_tau)}" if r.fitted_tau is not None else f"{f(r.default_tau):>11}"
            dev = (f"{r.dev_default.f1:.2f}->{r.dev_fitted.f1:.2f}" if r.dev_fitted else f"{r.dev_default.f1:11.2f}")
            td = f"{f(r.test_default.precision)}/{f(r.test_default.recall)}/{r.test_default.f1:.2f}"
            tf = (f"{f(r.test_fitted.precision)}/{f(r.test_fitted.recall)}/{r.test_fitted.f1:.2f}"
                  if r.test_fitted else "")
            lines.append(f"{r.rule:<22} {r.status:<18} {tau:>11}  {dev:>11}  {td:>19}  {tf:>19}  "
                         f"{f(r.brier)} {f(r.ece)} {f(r.auroc)}  {r.dev_pos}/{r.dev_neg}")
        return "\n".join(lines)


def calibrate(examples: list[Example], objective_name: str = "f1", min_positives: int = 5,
              min_negatives: int = 5, min_gain: float = 0.02) -> CalibrationReport:
    obj = objective(objective_name)
    report = CalibrationReport(objective=objective_name)
    for rule in sorted({e.rule for e in examples}):
        dev, test = _labelled(examples, rule, "dev"), _labelled(examples, rule, "test")
        allrows = dev + test
        dev_pos = sum(t for _, t, _ in dev)
        dev_neg = len(dev) - dev_pos
        base = dict(rule=rule, dev_pos=dev_pos, dev_neg=dev_neg, dev_default=_counts_at(dev, None),
                    test_default=_counts_at(test, None), brier=brier(test), ece=ece(test), auroc=auroc(allrows),
                    dev_fitted=None, test_fitted=None, fitted_tau=None)
        if not allrows:
            report.rules.append(RuleCalibration(status="no data", reason="no labelled units", default_tau=None, **base))
            continue
        if rule not in TUNABLE:
            report.rules.append(RuleCalibration(status="not tunable", default_tau=None,
                                                reason="verdict is not a single risk threshold", **base))
            continue
        default = _default_tau(rule)
        if dev_pos < min_positives or dev_neg < min_negatives:
            report.rules.append(RuleCalibration(status="insufficient data", default_tau=default,
                                                reason=f"dev has {dev_pos} positives / {dev_neg} negatives; "
                                                       f"need {min_positives}/{min_negatives}", **base))
            continue
        default_score = obj(_counts_at(dev, default))
        best = max(GRID, key=lambda t: (obj(_counts_at(dev, t)), -abs(t - default)))
        gain = obj(_counts_at(dev, best)) - default_score
        if best == default or gain < min_gain:
            report.rules.append(RuleCalibration(status="keep default", default_tau=default,
                                                reason=f"best dev gain {gain:+.3f} < {min_gain}", **base))
            continue
        base.update(fitted_tau=best, dev_fitted=_counts_at(dev, best), test_fitted=_counts_at(test, best))
        report.rules.append(RuleCalibration(status="tuned", default_tau=default, reason=f"dev gain {gain:+.3f}",
                                            override={"threshold": _param(rule, best)}, **base))
    return report
