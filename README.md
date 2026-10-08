# evaltrix

Agent governance and evals, judged by [Jev](https://typesafe.ai). Framework-agnostic: rules read a normalized
trace (OpenAI or Anthropic messages, single- or multi-agent), so any agent's logs work.

- **20 rules** across security, privacy, action safety, honesty and reliability, each tagged with the standards it
  produces evidence for (OWASP LLM Top 10 2025, OWASP Agentic AI threats, EU AI Act Art. 14, HIPAA, NIST AI 600-1).
- **One Jev request per unit** (trace, tool call, tool result, sub-agent): all rules for a unit are packed together.
- **Governance layer**: YAML policy packs, a hash-chained audit log, redacted evidence, an HTML report.
- **Measured**: a 73-trace labelled benchmark; test split P 0.87 / R 0.95 / F1 0.90 on jev-1.13.0
  (see `benchmark/README.md` for method, weaknesses and what is not measured).

```bash
pip install -e .
export TYPESAFE_API_KEY=...

evaltrix rules                                              # the catalog
evaltrix policy check examples/policies/agents.yaml         # validate a policy pack, see its digest
evaltrix observe traces.jsonl --policy examples/policies/agents.yaml --out out/
evaltrix verify out/audit.jsonl                             # tamper check
evaltrix bench labelled.jsonl --out preds.jsonl             # precision/recall on your labelled traces
evaltrix calibrate preds.jsonl --write overrides.yaml       # fit thresholds on dev, report on test
```

`observe` exits 1 when findings reach the policy's `fail_on` severity, so it can gate CI. Add `--fake` to any
command that calls Jev to check wiring without an API key.

Traces are JSONL, one per line: `{"trace_id", "messages", "tools"?, "agents"?, "context"?, ...metadata}`.
Multi-agent: mark assistant messages with `"agent"`, and delegation as
`{"role": "handoff", "from": "orchestrator", "to": "billing", "content": "<task or report>"}`.
Labelled files add `"labels": {"rule@unit": "fail"}` and `"split": "dev" | "test"`.

Calibration only recommends a threshold when dev has enough examples (5 positives and 5 negatives per rule by
default) and the gain is real; on small label sets it keeps the defaults and says why.

See `PLAN.md` for design decisions and what each milestone found.
