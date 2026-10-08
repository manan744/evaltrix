"""Policy packs: which rules run for a deployment, how strict they are, and who owns each finding.

    pack = PolicyPack.load("policies/support-agents.yaml")
    pack.rules        # instantiated rules, ready for evaluate()
    pack.digest       # sha256 of the resolved policy; changes whenever anything that affects results changes

YAML:

    pack: support-agents
    version: 2026.10.0
    owner: ai-governance@example.com
    mode: observe                  # observe: record only. (enforce is reserved for gates, later)
    profile: baseline              # starting rule set: baseline | strict | none
    rules:                         # per-rule overrides, keyed by rule name
      tool_risk:     {args: {allow_at: 0.85}, severity: critical, owner: platform-team}
      goal_completion: {}          # listing a rule enables it with defaults
      run_stats:     {args: {max_tool_calls: 25}}
      prompt_injection: {enabled: false}
      grounded:      {frameworks: ["internal:answer-quality-v2"]}   # extra tags, added to the rule's own
    record:
      metadata: [timestamp, agent_version, environment]   # trace metadata copied into audit records
      redact_evidence: true        # strip personal data and secrets from stored evidence (default)
    fail_on: high                  # lowest severity that makes a CI run fail (used by the CLI)
"""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .. import __version__
from ..evals.base import Eval
from ..evals.builtin import ALL_RULES
from ..evals.profiles import PROFILES

SEVERITIES = ("critical", "high", "medium", "low")
_RULES = {c.name: c for c in ALL_RULES}
_RULE_KEYS = {"enabled", "args", "severity", "owner", "frameworks"}
_TOP_KEYS = {"pack", "version", "owner", "mode", "profile", "rules", "record", "fail_on", "description"}


class PolicyError(ValueError):
    pass


@dataclass
class RuleConfig:
    rule: Eval
    severity: str
    owner: str
    frameworks: tuple[str, ...]
    args: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.rule.name


@dataclass
class PolicyPack:
    name: str
    version: str
    owner: str
    mode: str
    profile: str
    configs: list[RuleConfig]
    record_metadata: list[str]
    redact_evidence: bool
    fail_on: str | None
    digest: str
    source: dict[str, Any]

    @property
    def rule_fingerprints(self) -> dict[str, str]:
        """sha256 of the code that defines each enabled rule: its module (question wording, option
        descriptions, decision rule) plus the shared helpers that build its state."""
        return {c.name: rule_fingerprint(type(c.rule)) for c in self.configs}

    @property
    def rules(self) -> list[Eval]:
        return [c.rule for c in self.configs]

    def config(self, rule_name: str) -> RuleConfig:
        return next(c for c in self.configs if c.name == rule_name)

    # ------------------------------------------------------------------ loading

    @classmethod
    def load(cls, path: str | Path) -> PolicyPack:
        try:
            spec = yaml.safe_load(Path(path).read_text())
        except yaml.YAMLError as e:
            raise PolicyError(f"{path}: not valid YAML: {e}") from e
        return cls.from_dict(spec, where=str(path))

    @classmethod
    def from_dict(cls, spec: dict[str, Any], where: str = "policy") -> PolicyPack:
        if not isinstance(spec, dict):
            raise PolicyError(f"{where}: expected a mapping at the top level")
        unknown = set(spec) - _TOP_KEYS
        if unknown:
            raise PolicyError(f"{where}: unknown keys {sorted(unknown)}; allowed: {sorted(_TOP_KEYS)}")
        for key in ("pack", "version"):
            if not spec.get(key):
                raise PolicyError(f"{where}: `{key}` is required")
        mode = spec.get("mode", "observe")
        if mode != "observe":
            raise PolicyError(f"{where}: mode {mode!r} is not supported yet (only observe)")

        profile = spec.get("profile", "baseline")
        if profile != "none" and profile not in PROFILES:
            raise PolicyError(f"{where}: profile must be one of {sorted(PROFILES) + ['none']}, got {profile!r}")
        base: dict[str, Eval] = {} if profile == "none" else {r.name: r for r in PROFILES[profile]()}

        overrides = spec.get("rules") or {}
        if not isinstance(overrides, dict):
            raise PolicyError(f"{where}: `rules` must map rule names to settings")
        configs: dict[str, RuleConfig] = {}
        order = list(base) + [n for n in overrides if n not in base]
        for name in order:
            o = overrides.get(name) or {}
            if name not in _RULES:
                raise PolicyError(f"{where}: unknown rule {name!r}; known: {sorted(_RULES)}")
            bad = set(o) - _RULE_KEYS
            if bad:
                raise PolicyError(f"{where}: rules.{name}: unknown settings {sorted(bad)}; allowed: {sorted(_RULE_KEYS)}")
            if o.get("enabled", True) is False:
                continue
            cls_ = _RULES[name]
            args = dict(o.get("args") or {})
            if args:
                accepted = set(inspect.signature(cls_.__init__).parameters) - {"self"}
                if set(args) - accepted:
                    raise PolicyError(f"{where}: rules.{name}.args: unknown {sorted(set(args) - accepted)}; "
                                      f"accepted: {sorted(accepted)}")
                rule = cls_(**{**_profile_args(base.get(name)), **args})
            else:
                rule = base.get(name) or cls_()
            severity = o.get("severity", cls_.severity)
            if severity not in SEVERITIES:
                raise PolicyError(f"{where}: rules.{name}.severity must be one of {SEVERITIES}")
            configs[name] = RuleConfig(rule=rule, severity=severity, owner=o.get("owner", spec.get("owner", "")),
                                       frameworks=tuple(dict.fromkeys([*cls_.frameworks, *(o.get("frameworks") or [])])),
                                       args=_profile_args(rule))
        if not configs:
            raise PolicyError(f"{where}: no rules enabled")

        record = spec.get("record") or {}
        fail_on = spec.get("fail_on")
        if fail_on is not None and fail_on not in SEVERITIES:
            raise PolicyError(f"{where}: fail_on must be one of {SEVERITIES}")
        pack = cls(
            name=spec["pack"], version=str(spec["version"]), owner=spec.get("owner", ""), mode=mode, profile=profile,
            configs=list(configs.values()), record_metadata=list(record.get("metadata") or []),
            redact_evidence=bool(record.get("redact_evidence", True)), fail_on=fail_on, digest="", source=spec,
        )
        pack.digest = pack._compute_digest()
        return pack

    def _compute_digest(self) -> str:
        """Everything that changes what gets judged or how: rule set, every rule's effective arguments,
        severities and tags, recording settings, and a fingerprint of each rule's code, because question
        wording lives in code and changing it changes verdicts."""
        resolved = {
            "pack": self.name, "version": self.version, "mode": self.mode, "evaltrix": __version__,
            "rules": [{"name": c.name, "args": c.args, "severity": c.severity, "owner": c.owner,
                       "frameworks": list(c.frameworks), "code": rule_fingerprint(type(c.rule))}
                      for c in self.configs],
            "record_metadata": self.record_metadata, "redact_evidence": self.redact_evidence, "fail_on": self.fail_on,
        }
        return hashlib.sha256(json.dumps(resolved, sort_keys=True, default=str).encode()).hexdigest()


def rule_fingerprint(cls: type[Eval]) -> str:
    from .. import detect, text
    from ..evals import _common, base

    h = hashlib.sha256()
    for mod in (inspect.getmodule(cls), _common, base, detect, text):
        h.update(inspect.getsource(mod).encode())  # type: ignore[arg-type]
    return h.hexdigest()


def _profile_args(rule: Eval | None) -> dict[str, Any]:
    """The constructor arguments a rule instance runs with. Built-in rules keep every constructor
    argument as an attribute of the same name (tests/test_governance.py enforces it)."""
    if rule is None:
        return {}
    params = [p for p in inspect.signature(type(rule).__init__).parameters if p != "self"]
    return {p: getattr(rule, p) for p in params if hasattr(rule, p)}
