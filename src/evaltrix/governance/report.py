"""A single self-contained HTML report for a governance run."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Iterable

from ..trace.model import Trace
from ..trace.normalize import to_trace
from .observer import SEVERITY_RANK, Record, Run, _redact, group_by_trace
from .policy import SEVERITIES, PolicyPack

CSS = """
:root{--bg:#f6f6f3;--panel:#fff;--ink:#1c1c1a;--muted:#6b6a64;--line:#e3e2dc;--chip:#efeee8;
--critical:#b42318;--high:#c4540a;--medium:#9a7400;--low:#55606e;--ok:#1f7a4d;--bar:#1f7a4d}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#141413;--panel:#1d1d1b;--ink:#ecebe5;
--muted:#9b9a92;--line:#302f2b;--chip:#292825;--critical:#ff6b5e;--high:#ff9a4d;--medium:#e3bd3a;--low:#9aa6b5;
--ok:#4cc38a;--bar:#4cc38a}}
:root[data-theme=dark]{--bg:#141413;--panel:#1d1d1b;--ink:#ecebe5;--muted:#9b9a92;--line:#302f2b;--chip:#292825;
--critical:#ff6b5e;--high:#ff9a4d;--medium:#e3bd3a;--low:#9aa6b5;--ok:#4cc38a;--bar:#4cc38a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1120px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:13px;margin:36px 0 12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}
.meta{color:var(--muted);font-size:13px}code{font:12px ui-monospace,Menlo,monospace}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;margin-top:20px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.kpi b{display:block;font-size:26px;font-variant-numeric:tabular-nums}.kpi span{color:var(--muted);font-size:12px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;overflow-x:auto}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);font-weight:600}tr:last-child td{border-bottom:0}.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.rules table{min-width:820px}
.sev{display:inline-block;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.04em;padding:0 7px;
border-radius:99px;border:1px solid currentColor;white-space:nowrap}
.critical{color:var(--critical)}.high{color:var(--high)}.medium{color:var(--medium)}.low{color:var(--low)}.ok{color:var(--ok)}
.chip{display:inline-block;background:var(--chip);border-radius:6px;padding:0 6px;margin:1px 3px 1px 0;font-size:11px;color:var(--muted)}
.bar{height:6px;background:var(--chip);border-radius:3px;overflow:hidden;width:80px;display:inline-block;vertical-align:middle;margin-right:6px}
.bar i{display:block;height:100%;background:var(--bar)}
details.trace{background:var(--panel);border:1px solid var(--line);border-radius:10px;margin-bottom:10px}
details.trace>summary{cursor:pointer;padding:12px 16px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;list-style:none}
details.trace>summary::-webkit-details-marker{display:none}
.tid{font:12px ui-monospace,Menlo,monospace;color:var(--muted)}
.body{padding:0 16px 16px;display:grid;gap:12px}
.finding{border-left:3px solid var(--line);padding:4px 0 4px 12px}
.finding.critical{border-color:var(--critical)}.finding.high{border-color:var(--high)}
.finding.medium{border-color:var(--medium)}.finding.low{border-color:var(--low)}
.finding .t{font-weight:600}.finding .d{color:var(--muted);font-size:13px;word-break:break-word}
.convo{font-size:13px;border:1px solid var(--line);border-radius:8px;padding:10px 12px;display:grid;gap:6px;max-height:360px;overflow:auto}
.convo div{word-break:break-word}.role{font-weight:700;font-size:11px;text-transform:uppercase;color:var(--muted);margin-right:6px}
.handoff{color:var(--muted);font-style:italic}
.note{margin-top:28px;color:var(--muted);font-size:12px}
@media (max-width:640px){main{padding:20px 16px 48px}.kpi b{font-size:22px}}
"""


def _e(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _sev(s: str) -> str:
    return f'<span class="sev {s}">{s}</span>'


def _bar(rate: float | None) -> str:
    if rate is None:
        return '<span class="meta">—</span>'
    return f'<span class="bar"><i style="width:{rate * 100:.0f}%"></i></span>{rate:.0%}'


def _msg(m: Any, redact: bool) -> str:
    def clean(s: str) -> str:
        s = _redact(s) if redact else s
        return s if len(s) <= 500 else s[:500] + "…"

    who = f"{m.role}" + (f" · {m.agent}" if m.agent else "")
    if m.role == "handoff":
        return f'<div class="handoff"><span class="role">handoff</span>{_e(m.agent)} → {_e(m.to)}: {_e(clean(m.content))}</div>'
    if m.tool_calls:
        calls = ", ".join(f"{c.name}({json.dumps(c.args)})" for c in m.tool_calls)
        return f'<div><span class="role">{_e(who)}</span>→ {_e(clean(calls))}</div>'
    return f'<div><span class="role">{_e(who)}</span>{_e(clean(m.content))}</div>'


def render(run: Run, pack: PolicyPack, path: str | Path, traces: Iterable[Any] | None = None,
           title: str = "Agent governance report") -> Path:
    s = run.summary()
    by_trace = group_by_trace(run.findings)
    normalized: dict[str | None, Trace] = {}
    for t in traces or []:
        tt = to_trace(t)
        normalized[tt.id] = tt
    redact = pack.redact_evidence

    decisions = sum(v for row in s["by_rule"] for k, v in row["decisions"].items() if k in ("escalate", "block"))
    kpis = [
        (s["traces"], "traces observed"),
        (s["traces_with_findings"], f"with findings ({s['traces_with_findings'] / max(1, s['traces']):.0%})"),
        (s["by_severity"]["critical"], "critical findings"),
        (s["by_severity"]["high"], "high findings"),
        (decisions, "tool calls a gate would have stopped"),
        (s["errors"], "coverage gaps (rule errors)"),
        (f"${run.usage.cost_usd:.4f}", f"{run.usage.requests} Jev requests · {run.usage.input_tokens:,} tokens"),
    ]
    kpi_html = "".join(f'<div class="kpi"><b>{_e(v)}</b><span>{_e(k)}</span></div>' for v, k in kpis)

    def count_table(title_: str, counts: dict[str, int], empty: str) -> str:
        if not counts:
            return f'<div class="panel"><table><tr><th>{_e(title_)}</th></tr><tr><td class="meta">{_e(empty)}</td></tr></table></div>'
        rows = "".join(f"<tr><td>{_e(k)}</td><td class='num'>{v}</td></tr>" for k, v in counts.items())
        return f'<div class="panel"><table><tr><th>{_e(title_)}</th><th>findings</th></tr>{rows}</table></div>'

    breakdowns = count_table("Standard", s["by_framework"], "No findings") + count_table(
        "Category", s["by_category"], "No findings")
    if any(t.get("agents") for t in run.traces):
        breakdowns += count_table("Agent", s["by_agent"], "No agent-level findings")

    rule_rows = []
    for row in sorted(s["by_rule"], key=lambda r: (-SEVERITY_RANK[r["severity"]], r["rule"])):
        chips = " ".join(f'<span class="chip">{_e(k)}: {v}</span>' for k, v in row["decisions"].items())
        gaps = (f' <span class="chip">{row["skipped"]} n/a</span>' if row["skipped"] else "") + (
            f' <span class="chip critical">{row["errors"]} errors</span>' if row["errors"] else "")
        tags = "".join(f'<span class="chip">{_e(t)}</span>' for t in row["frameworks"])
        rule_rows.append(
            f"<tr><td><b>{_e(row['rule'])}</b><br><span class='meta'>{_e(row['category'])}</span></td>"
            f"<td>{_sev(row['severity'])}</td><td class='num'>{_bar(row['pass_rate'])}</td>"
            f"<td class='num'>{row['findings']} / {row['evaluated']}{gaps}</td><td>{chips}</td>"
            f"<td class='meta'>{_e(row['owner'])}<br>{tags}</td></tr>")

    trace_html = []
    order = sorted(run.traces, key=lambda t: -max((SEVERITY_RANK[r.severity] for r in by_trace.get(t["trace_id"], [])),
                                                  default=0))
    for t in order:
        fs: list[Record] = sorted(by_trace.get(t["trace_id"], []), key=lambda r: -SEVERITY_RANK[r.severity])
        worst = fs[0].severity if fs else None
        badge = _sev(worst) if worst else '<span class="sev ok">clean</span>'
        items = "".join(
            f'<div class="finding {r.severity}"><div class="t">{_e(r.rule)} <span class="meta">@ {_e(r.target)}'
            f'{" · agent " + _e(r.agent) if r.agent else ""}</span></div>'
            f'<div class="d">{_sev(r.severity)}{" · would " + _e(r.decision) if r.decision in ("escalate", "block") else ""}'
            f'{f" · p={r.p:.2f}" if r.p is not None else ""}<br>{_e(r.detail)}</div></div>' for r in fs)
        tr = normalized.get(t["trace_id"])
        convo = f'<div class="convo">{"".join(_msg(m, redact) for m in tr.messages if m.role != "system")}</div>' if tr else ""
        meta = " · ".join(f"{_e(k)} {_e(v)}" for k, v in t.items() if k not in ("trace_id", "agents", "first_user_message"))
        agents = f" · agents: {_e(', '.join(t['agents']))}" if t.get("agents") else ""
        trace_html.append(
            f'<details class="trace"{" open" if worst == "critical" else ""}><summary>{badge}'
            f'<span>{_e(t.get("first_user_message", ""))[:120]}</span>'
            f'<span class="tid">{_e(t["trace_id"])}{" · " + meta if meta else ""}{agents} · {len(fs)} finding{"s" * (len(fs) != 1)}'
            f'</span></summary><div class="body">{items or "<div class=meta>No findings.</div>"}{convo}</div></details>')

    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Governance Report</title><style>{CSS}</style></head>
<body><main>
<h1>{_e(title)}</h1>
<div class="meta">Policy <b>{_e(pack.name)}</b> v{_e(pack.version)} · profile {_e(pack.profile)} · mode <b>{_e(pack.mode)}</b>
· digest <code>{_e(pack.digest[:12])}</code> · run <code>{_e(run.run_id)}</code> · {_e(run.started_at[:19].replace("T", " "))} UTC
· judge {_e(", ".join(run.models) or "n/a")}</div>
<div class="kpis">{kpi_html}</div>
<h2>Where the findings are</h2><div class="grid2">{breakdowns}</div>
<h2>Rules</h2>
<div class="panel rules"><table><tr><th>Rule</th><th>Severity</th><th>Pass rate</th><th>Findings</th><th>Decisions</th>
<th>Owner · evidence for</th></tr>{"".join(rule_rows)}</table></div>
<h2>Traces</h2>{"".join(trace_html)}
<p class="note">Observe mode: nothing was blocked. "Would escalate/block" is what the tool-risk decision would do if enforced.
Coverage gaps are rules that errored and therefore judged nothing. Every record is in the hash-chained audit log with
the policy digest above.{" Personal data and secrets are redacted in this report." if redact else ""}</p>
</main></body></html>"""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc)
    return out
