"""The benchmark's labels must be well-formed: a typo in a label would silently turn a positive into a negative."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from evaltrix import Trace
from evaltrix.evals import ALL_RULES
from evaltrix.trace import units

BENCH = Path(__file__).parents[1] / "benchmark"
sys.path.insert(0, str(BENCH))

from build_dataset import TRACES, split_of  # noqa: E402
from evaltrix.labelled import Example, score  # noqa: E402

RULES = {c.name: c for c in ALL_RULES}


def test_file_matches_builder():
    rows = [json.loads(line) for line in (BENCH / "traces.jsonl").read_text().splitlines() if line.strip()]
    assert [r["id"] for r in rows] == [t["id"] for t in TRACES], "run benchmark/build_dataset.py"
    assert [r["labels"] for r in rows] == [t["labels"] for t in TRACES], "run benchmark/build_dataset.py"
    assert all(r["split"] == split_of(r["id"]) for r in rows)


def test_every_label_points_at_a_real_unit():
    for t in TRACES:
        trace = Trace.from_any(t["trace"])
        keys = {u.key: u.kind for u in units(trace)}
        for label, value in t["labels"].items():
            rule, _, target = label.partition("@")
            assert rule in RULES, (t["id"], label)
            assert target in keys, (t["id"], label, sorted(keys))
            assert keys[target] == RULES[rule].target, (t["id"], label, "wrong unit kind for this rule")
            if rule == "tool_risk":
                assert value == "*" or set(value.split("|")) <= {"allow", "escalate", "block"}, (t["id"], label)
            else:
                assert value in ("fail", "*"), (t["id"], label)


def test_benchmark_covers_every_scored_rule_with_positives_and_hard_negatives():
    positives = Counter()
    for t in TRACES:
        for label, value in t["labels"].items():
            rule = label.split("@")[0]
            if value == "fail" or (rule == "tool_risk" and value != "*" and "allow" not in value):
                positives[rule] += 1
    # harmful_output deliberately ships with negatives only: positives must come from a vetted external dataset
    missing = {r for r in RULES if r not in ("run_stats", "harmful_output")} - set(positives)
    assert not missing, f"rules with no positive example: {missing}"
    assert sum("HARD NEGATIVE" in t["about"] for t in TRACES) >= 8
    assert sum(t["multi_agent"] for t in TRACES) >= 8


def test_scoring_semantics():
    def ex(rule, label, status, decision=None):
        return Example("x", "dev", rule, "trace", label, status, decision, None, None, "")

    s = score([ex("pii", "fail", "fail"), ex("pii", "fail", "skipped"), ex("pii", "pass", "fail"),
               ex("pii", "pass", "pass"), ex("pii", "*", "fail"), ex("pii", "fail", "error")])["pii"]
    assert (s.tp, s.fn, s.fp, s.tn, s.errors) == (1, 1, 1, 1, 1)  # skipped positive = miss; * excluded
    t = score([ex("tool_risk", "escalate|block", "fail", "block"), ex("tool_risk", "allow|escalate", "fail", "escalate"),
               ex("tool_risk", "allow", "fail", "escalate"), ex("tool_risk", "escalate", "fail", "block")])["tool_risk"]
    assert (t.tp, t.fp, t.fn) == (2, 1, 0)  # allow|escalate excluded from detection
    assert (t.decisions_ok, t.decisions_total) == (2, 4)
