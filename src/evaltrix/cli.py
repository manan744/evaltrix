"""evaltrix command line.

    evaltrix rules                                     list the rule catalog
    evaltrix policy check POLICY.yaml                  validate a policy pack, show resolved rules and digest
    evaltrix observe TRACES.jsonl --policy P.yaml      judge traces; append to the audit log; write the HTML report
    evaltrix verify AUDIT.jsonl                        check the audit log's hash chain
    evaltrix bench LABELLED.jsonl [...]                run rules on labelled traces; P/R per rule; save predictions
    evaltrix bench --from PREDICTIONS.jsonl            re-score saved predictions without calling Jev
    evaltrix calibrate PREDICTIONS.jsonl [...]         fit thresholds on dev, report on test, write overrides

Jev is called with TYPESAFE_API_KEY from the environment. `--fake` runs the wiring with neutral answers.
Exit codes: 0 ok; 1 findings at/above the policy's fail_on (observe) or a broken chain (verify); 2 usage errors.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _jev(fake: bool):
    from .jev import FakeJev, Jev, JevError

    if fake:
        return FakeJev()
    try:
        return Jev()
    except JevError as e:
        raise SystemExit(f"error: {e} (or pass --fake)") from None


def _read_jsonl(path: str) -> list[dict[str, Any]]:
    try:
        lines = Path(path).read_text().splitlines()
    except OSError as e:
        raise SystemExit(f"error: {e}") from None
    rows = []
    for n, line in enumerate(lines, 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"error: {path}:{n}: not JSON ({e.msg})") from None
    return rows


# --------------------------------------------------------------------------- commands


def cmd_rules(a: argparse.Namespace) -> int:
    from .evals import ALL_RULES, baseline

    in_baseline = {type(r).name for r in baseline()}
    if a.json:
        print(json.dumps([{"name": c.name, "target": c.target, "category": c.category, "severity": c.severity,
                           "baseline": c.name in in_baseline, "frameworks": list(c.frameworks),
                           "description": (c.__doc__ or "").strip().splitlines()[0]} for c in ALL_RULES], indent=2))
        return 0
    print(f"{'rule':<22} {'target':<12} {'category':<14} {'severity':<9} {'base':<5} description")
    for c in ALL_RULES:
        doc = (c.__doc__ or "").strip().splitlines()[0]
        print(f"{c.name:<22} {c.target:<12} {c.category:<14} {c.severity:<9} {'yes' if c.name in in_baseline else '':<5} "
              f"{doc[:80]}")
    return 0


def cmd_policy_check(a: argparse.Namespace) -> int:
    from .governance import PolicyError, PolicyPack

    try:
        pack = PolicyPack.load(a.policy)
    except (PolicyError, OSError) as e:
        print(f"invalid: {e}", file=sys.stderr)
        return 2
    print(f"ok  {pack.name} v{pack.version} · profile {pack.profile} · mode {pack.mode} · digest {pack.digest[:16]}")
    print(f"    fail_on={pack.fail_on} · redact_evidence={pack.redact_evidence} · metadata={pack.record_metadata}")
    for c in pack.configs:
        args = ", ".join(f"{k}={v}" for k, v in c.args.items() if v is not None)
        print(f"    {c.name:<22} {c.severity:<9} owner={c.owner or '-':<24} {args}")
    return 0


async def _observe(a: argparse.Namespace) -> int:
    from .governance import AuditLog, PolicyError, PolicyPack, observe, render

    try:
        pack = PolicyPack.load(a.policy)
    except (PolicyError, OSError) as e:
        print(f"invalid policy: {e}", file=sys.stderr)
        return 2
    traces = _read_jsonl(a.traces)
    jev = _jev(a.fake)
    run = await observe(traces, pack, jev=jev, concurrency=a.concurrency)
    if hasattr(jev, "aclose"):
        await jev.aclose()
    out = Path(a.out)
    head = AuditLog(out / "audit.jsonl").append(run)
    report = render(run, pack, out / "governance_report.html", traces=traces)
    s = run.summary()
    print(f"{pack.name} v{pack.version} · {s['traces']} traces · {run.usage}")
    print(f"findings {s['findings']} in {s['traces_with_findings']} traces · "
          + " · ".join(f"{v} {k}" for k, v in s["by_severity"].items()) + f" · coverage gaps {s['errors']}")
    print(f"audit  {out / 'audit.jsonl'} (head {head[:12]})\nreport {report}")
    fail_on = a.fail_on or pack.fail_on
    if fail_on and run.fails(fail_on):
        print(f"FAIL: findings at or above {fail_on}")
        return 1
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from .governance import AuditLog

    ok, _, msg = AuditLog(a.audit).verify()
    print(("ok    " if ok else "FAIL  ") + msg)
    return 0 if ok else 1


def _score_table(examples, split: str) -> str:
    from .labelled import micro, score

    s = score(examples, split)

    def f(x):
        return "  -  " if x is None else f"{x:5.2f}"

    lines = [f"== {split} ==", f"{'rule':<22} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>4}  {'prec':>5} {'recall':>6} {'F1':>5}"]
    for name in sorted(s):
        r = s[name]
        lines.append(f"{name:<22} {r.tp:>3} {r.fp:>3} {r.fn:>3} {r.tn:>4}  {f(r.precision)} {f(r.recall):>6} {f(r.f1)}"
                     + (f"  errors {r.errors}" if r.errors else ""))
    m = micro(s)
    lines.append(f"{'ALL (micro)':<22} {m.tp:>3} {m.fp:>3} {m.fn:>3} {m.tn:>4}  {f(m.precision)} {f(m.recall):>6} {f(m.f1)}")
    return "\n".join(lines)


def _print_scores(examples, show_mistakes: bool) -> None:
    from .labelled import score

    for split in ("dev", "test"):
        if not any(e.split == split for e in examples):
            continue
        print(_score_table(examples, split) + "\n")
        if show_mistakes:
            for rs in score(examples, split).values():
                for e in rs.mistakes:
                    got = e.decision if e.rule == "tool_risk" else e.status
                    print(f"  {'FN' if e.positive else 'FP'} {e.trace_id:<6} {e.rule}@{e.target:<34} "
                          f"label={e.label:<14} got={got:<9} {e.detail[:90]}")
            print()


async def _bench(a: argparse.Namespace) -> int:
    from .engine import evaluate_many
    from .evals import ALL_RULES
    from .governance import PolicyPack
    from .jev import Usage
    from .labelled import examples_from, load_labelled, load_predictions, save_predictions

    if a.from_file:
        try:
            examples = load_predictions(a.from_file)
        except (ValueError, OSError, TypeError) as e:
            raise SystemExit(f"error: {e}") from None
        _print_scores(examples, a.mistakes)
        return 0
    if not a.labelled:
        raise SystemExit("error: give labelled trace files, or --from saved predictions")
    try:
        items = load_labelled(a.labelled)
    except (ValueError, OSError) as e:
        raise SystemExit(f"error: {e}") from None
    if a.policy:
        rules = PolicyPack.load(a.policy).rules
    else:
        rules = [c() for c in ALL_RULES if not a.rules or c.name in a.rules.split(",")]
    jev = _jev(a.fake)
    reports = await evaluate_many([i["trace"] for i in items], rules, jev=jev, concurrency=a.concurrency)
    if hasattr(jev, "aclose"):
        await jev.aclose()
    examples = [e for item, rep in zip(items, reports) for e in examples_from(item, rep)]
    usage = Usage()
    for r in reports:
        usage = usage + r.usage
    out = Path(a.out or f"predictions_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.jsonl")
    save_predictions(examples, out)
    print(f"{len(items)} traces · {len(examples)} scored units · {usage}\npredictions {out}\n")
    _print_scores(examples, a.mistakes)
    return 0


def cmd_calibrate(a: argparse.Namespace) -> int:
    from .calibrate import calibrate
    from .labelled import load_predictions

    try:
        examples = load_predictions(a.predictions)
        report = calibrate(examples, a.objective, a.min_positives, a.min_negatives, a.min_gain)
    except (ValueError, OSError, TypeError) as e:
        raise SystemExit(f"error: {e}") from None
    print(report.table())
    notes = [f"  {r.rule}: {r.reason}" for r in report.rules if r.status in ("insufficient data", "keep default")]
    if notes:
        print("\nnot tuned:\n" + "\n".join(notes))
    overrides = report.policy_overrides()
    tuned = list(overrides["rules"])
    print(f"\ntuned: {', '.join(tuned) if tuned else 'none'}")
    if a.write:
        Path(a.write).write_text("# merge into your policy pack's `rules:` (from evaltrix calibrate)\n"
                                 + yaml.safe_dump(overrides, sort_keys=True))
        print(f"overrides written to {a.write}")
    return 0


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="evaltrix", description="Agent governance with Jev.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("rules", help="list the rule catalog")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_rules)

    p = sub.add_parser("policy", help="policy pack tools")
    psub = p.add_subparsers(dest="policy_command", required=True)
    pc = psub.add_parser("check", help="validate a policy pack")
    pc.add_argument("policy")
    pc.set_defaults(fn=cmd_policy_check)

    p = sub.add_parser("observe", help="judge traces under a policy pack")
    p.add_argument("traces")
    p.add_argument("--policy", required=True)
    p.add_argument("--out", default="evaltrix-out")
    p.add_argument("--fail-on", choices=["critical", "high", "medium", "low"])
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--fake", action="store_true")
    p.set_defaults(afn=_observe)

    p = sub.add_parser("verify", help="check an audit log's hash chain")
    p.add_argument("audit")
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("bench", help="run rules on labelled traces and score them")
    p.add_argument("labelled", nargs="*")
    p.add_argument("--from", dest="from_file", nargs="+", help="re-score saved predictions instead of running")
    p.add_argument("--mistakes", action="store_true", help="list every false positive / false negative")
    p.add_argument("--rules", help="comma-separated rule names (default: all)")
    p.add_argument("--policy", help="use a policy pack's rules and settings instead of --rules")
    p.add_argument("--out", help="predictions file (default: predictions_<time>.jsonl)")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--fake", action="store_true")
    p.set_defaults(afn=_bench)

    p = sub.add_parser("calibrate", help="fit thresholds on dev predictions, report on test")
    p.add_argument("predictions", nargs="+", help="prediction files; later ones override earlier for the same unit")
    p.add_argument("--objective", default="f1", help="f1 | recall@precision>=0.9 | precision@recall>=0.9")
    p.add_argument("--min-positives", type=int, default=5)
    p.add_argument("--min-negatives", type=int, default=5)
    p.add_argument("--min-gain", type=float, default=0.02)
    p.add_argument("--write", help="write recommended overrides as YAML")
    p.set_defaults(fn=cmd_calibrate)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if hasattr(a, "afn"):
        return asyncio.run(a.afn(a))
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
