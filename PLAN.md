# evaltrix: plan

Evals and governance for agent traces, using Jev (TypeSafe) as the judge. Inspired by the design of
[jevals](https://github.com/openlayer-ai/jevals) (MIT); the code here is written from scratch.

## How Jev is used

Jev is a decision model, not a chat model. One request carries a JSON `state` and a set of typed
questions, and every question comes back as a probability from a single forward pass:

| type | we send | Jev returns |
|---|---|---|
| yes/no (`noul` on the wire) | `{type, instructions, criteria?: {true, false}}` | P(true) |
| choice | `{type, instructions, criteria: {option: description}}` | probability per option |
| score | `{type, instructions, criteria: [lowest … highest]}` | probability per level |

`POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer $TYPESAFE_API_KEY`,
`X-Data-Retention: zero`. About $0.042 per million input tokens, no charge for output.

## Ideas we keep

1. **An eval is three functions.** `state()` chooses what Jev reads (keep it small, since input tokens
   are the cost), `questions()` asks typed questions, `reduce()` turns probabilities into a score,
   a pass/fail result and evidence.
2. **Code first, then Jev.** A deterministic precheck finds candidates (regex, detectors) and Jev only
   makes the judgement call.
3. **Packing.** All evals for one unit are merged into one request: states merged by key, question
   ids namespaced `<eval>.<qid>`, a separate request when two states conflict.
4. **One definition for everything.** The same eval works offline, in monitoring and as a gate.
5. **Calibration** against human labels before trusting a threshold.

## Changes from jevals

- **Targets are first-class.** Each eval says whether it runs on the whole trace, each tool call
  (with the conversation up to that point) or each tool result. jevals' trace-level injection check
  reads only the last tool result.
- **Strict defaults.** Grounding fails on any unsupported claim, and gates fail closed when Jev errors.
- **Usage per eval.** Each eval is charged its share of a packed request's input tokens.
- **The fake backend uses rules**, so tests check real behaviour instead of 0.5 everywhere.
- **One transport (TypeSafe)**, behind a small interface so another one can be added later.
- **Not in v1:** framework adapters and an MCP server.

## Layout

```
src/evaltrix/
  jev/        types.py (questions/answers) · client.py (TypeSafe HTTP) · fake.py (rule-based test backend)
  trace/      normalize.py (OpenAI / Anthropic → Trace) · targets.py (trace / tool_call / tool_result units)
  evals/      base.py · builtin/ · yaml.py (declarative evals + safe expressions)
  engine/     pack.py · run.py (evaluate, evaluate_many)
  governance/ policy pack · audit log (hash-chained) · findings · HTML report
  calibrate.py · cli.py
```

## Scope

Rules are about how agents behave (security, privacy, action safety, honesty, reliability), never about a
business domain, and read only the normalized trace, so they work for any agent framework. Multi-agent
traces are first-class: per-agent attribution, handoffs, each sub-agent judged against its delegated task.
Domain rules are for users to add themselves (milestone 7).

## Milestones

1. [x] Jev client and types. Confirmed live (jev-1.13.0): `model` is required (default `jev-latest`).
2. [x] Trace normalizer (OpenAI + Anthropic), targets, eval base, packing engine, per-eval usage.
3. [x] First built-in rules (grounded, indirect injection, tool risk, PII), validated live.
       Lesson: question wording is the main lever; a too-literal "args grounded" question escalated harmless searches.
4. [x] Governance catalog: 18 rules + `baseline` / `strict` profiles, OWASP LLM 2025 / OWASP Agentic / EU AI Act tags.
       Multi-agent traces (agents, handoffs, delegation vs return, `agent` and `handoff` targets), session `context`.
       Secrets are masked before Jev sees them. Live on 13 agent-behaviour scenarios (`tests/fixtures/agent_scenarios.py`):
       25/25 expected findings, 0 false positives, 0 errors, ~$0.003 for the full catalog run.
       Lessons: "block" must mean harmful, not redundant (repeated searches were blocked); session facts such as
       the signed-in user must be visible to tool-call rules or their arguments look ungrounded.
5. [x] Governance layer (`src/evaltrix/governance/`): YAML policy packs (profile + per-rule overrides of args,
       severity, owner, tags), a digest over the resolved policy *and each rule's code*, observe runs producing one
       record per rule x unit (agent, decision, severity, standards), evidence redaction on by default, a
       hash-chained audit log with per-run summary records, and a self-contained HTML report (by standard,
       category, agent, rule, trace). Now `evaltrix observe`; live: 25/25 expected, 0 extra, 0 gaps, $0.002/run.
       Lessons: (1) Jev is near-deterministic (+-0.01) and confident answers ignore request context, but
       borderline answers shift up to ~0.1 with what else is packed, so requests are now canonically ordered
       and borderline-prone questions are split into clear-cut steps (goal_hijacking: clean traces 0.45 -> 0.14).
       (2) A digest that ignores question wording lets different behaviour hide under the same policy id.
6. [x] Calibration (`evaltrix calibrate`): per-rule risk score, thresholds fitted on dev only and reported on test,
       guarded by minimum example counts and minimum gain; Brier, ECE, AUROC per rule; overrides written as a policy
       snippet. Command line (`evaltrix rules | policy check | observe | verify | bench | calibrate`). Two rules
       restated as single-score (oversight_bypass, tool_error_masking; same verdicts, re-run live to confirm).
       Consolidated live run, all 20 rules, 73 traces: test P 0.88 / R 0.95 / F1 0.91, $0.012. Calibration tunes
       nothing yet (no rule has 5 dev positives), but AUROC is 1.00 for most rules incl. tool_risk: the errors are
       where the line is drawn, not how Jev ranks, so tuning will pay off with more labelled data.
       Next candidates: make tool_risk tunable (one risk threshold), more labelled data per rule.
7. [ ] Custom rules in YAML (user-defined, domain-specific allowed) with a safe expression language.
8. [x] Benchmark (`benchmark/`): 65 labelled traces (9 agent kinds, 9 multi-agent incl. a 3-agent chain and a
       cascading hijack, 13 hard negatives), frozen labels, 40/60 dev/test split, unit-level P/R per rule, every
       prediction saved for calibration. Test (never tuned on): P 0.85 / R 0.93 / F1 0.88 after dev-only fixes
       (was 0.76 / 0.93 / 0.83). Known weaknesses and proposed label corrections are in benchmark/README.md.
   [x] Added rules: phi (health information about someone other than the user; 3/3 caught, 0 FP, own-results trap
       passes) and harmful_output (content that helps cause harm; 0 FP on 71 negatives, recall needs an external
       vetted dataset via `evaltrix bench <your file>`). Catalog: 20 rules; baseline 15.
