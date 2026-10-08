from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from evaltrix.calibrate import auroc, brier, calibrate, ece, objective
from evaltrix.cli import main
from evaltrix.labelled import Example

from .fixtures.agent_scenarios import SCENARIOS

EXAMPLE_POLICY = Path(__file__).parents[1] / "examples" / "policies" / "agents.yaml"


def ex(rule: str, split: str, positive: bool, p: float, flagged_at: float = 0.5, i: int = 0) -> Example:
    status = "fail" if (1 - p if rule == "goal_completion" else p) >= flagged_at else "pass"
    return Example(f"t{i}", split, rule, "trace", "fail" if positive else "pass", status, None, p, None, "")


def _set(rule: str, pos_p: list[float], neg_p: list[float], split: str) -> list[Example]:
    out = [ex(rule, split, True, p, i=i) for i, p in enumerate(pos_p)]
    return out + [ex(rule, split, False, p, i=100 + i) for i, p in enumerate(neg_p)]


# --------------------------------------------------------------------------- calibration maths


def test_metrics_on_known_values():
    rows = [(0.9, True, "fail"), (0.8, True, "fail"), (0.2, False, "pass"), (0.1, False, "pass")]
    assert auroc(rows) == 1.0
    assert brier(rows) == pytest.approx((0.01 + 0.04 + 0.04 + 0.01) / 4)
    assert ece([(0.9, True, ""), (0.9, False, "")]) == pytest.approx(0.4)  # says 90%, right 50% of the time
    assert auroc([(0.5, True, ""), (0.5, False, "")]) == 0.5
    with pytest.raises(ValueError):
        objective("accuracy")


def test_threshold_is_tuned_on_dev_and_reported_on_test():
    # positives sit at 0.35-0.45: the default 0.5 misses them all, 0.3 separates cleanly
    dev = _set("pii", [0.35, 0.4, 0.42, 0.45, 0.38], [0.05, 0.1, 0.02, 0.15, 0.2, 0.08], "dev")
    test = _set("pii", [0.36, 0.41, 0.44], [0.1, 0.12, 0.6], "test")
    (r,) = calibrate(dev + test).rules
    assert r.status == "tuned" and r.default_tau == 0.5 and r.fitted_tau is not None and r.fitted_tau <= 0.35
    assert r.dev_default.recall == 0 and r.dev_fitted.recall == 1.0
    assert r.test_fitted.tp == 3 and r.test_fitted.fp == 1  # the 0.6 negative is still a false positive
    assert r.override == {"threshold": r.fitted_tau}


def test_inverse_rules_map_back_to_their_own_argument():
    # goal_completion's saved p is p(ok); risk = 1 - p; its threshold argument is on the ok side
    dev = _set("goal_completion", [0.7, 0.65, 0.6, 0.62, 0.68], [0.95, 0.9, 0.97, 0.92, 0.99], "dev")
    (r,) = calibrate(dev).rules
    assert r.status == "tuned"
    assert r.override == {"threshold": round(1 - r.fitted_tau, 4)}


def test_small_samples_and_untunable_rules_keep_defaults():
    dev = _set("phi", [0.9, 0.95], [0.1] * 20, "dev")
    (r,) = calibrate(dev).rules
    assert r.status == "insufficient data" and r.override is None and "2 positives" in r.reason

    good = _set("pii", [0.9, 0.95, 0.8, 0.85, 0.99], [0.1, 0.05, 0.2, 0.15, 0.02], "dev")
    assert calibrate(good).rules[0].status == "keep default"  # already perfect: nothing to gain

    g = [Example("t", "dev", "grounded", "trace", "fail", "fail", None, 0.9, None, "")]
    assert calibrate(g).rules[0].status == "not tunable"


def test_recall_at_precision_objective():
    dev = _set("pii", [0.3, 0.4, 0.6, 0.7, 0.8], [0.35, 0.1, 0.2, 0.05, 0.15], "dev")
    (r,) = calibrate(dev, "recall@precision>=0.99").rules
    # precision 1.0 needs tau above the highest negative (0.35); 0.4 keeps 4 of 5 positives (default 0.5 keeps 3)
    assert r.status == "tuned" and r.fitted_tau == 0.4
    assert r.dev_fitted.precision == 1.0 and r.dev_fitted.recall == 0.8 and r.dev_default.recall == 0.6


# --------------------------------------------------------------------------- CLI


def _labelled_file(tmp_path: Path) -> Path:
    path = tmp_path / "labelled.jsonl"
    rows = [{"id": s["name"], "split": "dev" if i % 2 else "test", "trace": s["trace"], "labels": s["expected"]}
            for i, s in enumerate(SCENARIOS)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_cli_rules_and_policy(capsys):
    assert main(["rules", "--json"]) == 0
    rules = json.loads(capsys.readouterr().out)
    assert len(rules) == 20 and {"phi", "harmful_output"} <= {r["name"] for r in rules}
    assert main(["policy", "check", str(EXAMPLE_POLICY)]) == 0
    assert "digest" in capsys.readouterr().out


def test_cli_observe_verify_and_fail_on(tmp_path, capsys):
    traces = tmp_path / "traces.jsonl"
    traces.write_text("".join(json.dumps({"trace_id": s["name"], **s["trace"]}) + "\n" for s in SCENARIOS[:3]))
    out = tmp_path / "out"
    code = main(["observe", str(traces), "--policy", str(EXAMPLE_POLICY), "--out", str(out), "--fake"])
    assert code in (0, 1) and (out / "governance_report.html").exists()
    assert main(["verify", str(out / "audit.jsonl")]) == 0
    lines = (out / "audit.jsonl").read_text().splitlines()
    row = json.loads(lines[0])
    row["status"] = "tampered"
    (out / "audit.jsonl").write_text("\n".join([json.dumps(row), *lines[1:]]) + "\n")
    assert main(["verify", str(out / "audit.jsonl")]) == 1


def test_cli_bench_then_calibrate(tmp_path, capsys):
    labelled = _labelled_file(tmp_path)
    preds = tmp_path / "preds.jsonl"
    assert main(["bench", str(labelled), "--fake", "--out", str(preds)]) == 0
    assert preds.exists() and "ALL (micro)" in capsys.readouterr().out
    overrides = tmp_path / "overrides.yaml"
    assert main(["calibrate", str(preds), "--write", str(overrides)]) == 0
    out = capsys.readouterr().out
    assert "insufficient data" in out  # 13 scenarios is too few to tune anything, and it says so
    assert "rules" in yaml.safe_load(overrides.read_text())


def test_cli_errors_are_clean(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{not json\n")
    with pytest.raises(SystemExit, match="not JSON"):
        main(["observe", str(bad), "--policy", str(EXAMPLE_POLICY), "--fake"])
    with pytest.raises(SystemExit):
        main(["calibrate", str(tmp_path / "missing.jsonl")])


def test_risk_orientation_for_decision_and_code_rules():
    from evaltrix.calibrate import _risk

    allow = Example("t", "dev", "tool_risk", "tool_call[0]:x", "allow", "pass", "allow", 0.97, 0.97, "")
    block = Example("t", "dev", "tool_risk", "tool_call[1]:y", "block", "fail", "block", 0.9, 0.05, "")
    assert _risk(allow) == pytest.approx(0.03) and _risk(block) == pytest.approx(0.95)  # not the chosen option's p
    code = Example("t", "dev", "undeclared_tool", "tool_call[0]:x", "fail", "fail", None, None, 0.0, "")
    assert _risk(code) == 1.0
