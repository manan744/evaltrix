"""Append-only, hash-chained audit log (JSON lines).

Every line carries the hash of the line before it, so editing, reordering or deleting any past record
breaks the chain, and `verify()` says where. A run writes its evaluation records followed by one
`kind: "run"` record summarising it (pack, digest, counts, usage).

This makes tampering *evident*, not impossible: whoever can rewrite the file can rewrite the whole chain.
For that, ship the log (or the latest hash) somewhere the writer can't change: object storage with
retention, a SIEM, or a signed timestamp.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .observer import Run

GENESIS = "0" * 64


def _hash(prev: str, record: dict[str, Any]) -> str:
    body = json.dumps({k: v for k, v in record.items() if k not in ("prev_hash", "hash")}, sort_keys=True, default=str)
    return hashlib.sha256((prev + body).encode()).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def last_hash(self) -> str:
        if not self.path.exists():
            return GENESIS
        last = None
        with self.path.open() as f:
            for line in f:
                if line.strip():
                    last = line
        return json.loads(last)["hash"] if last else GENESIS

    def append(self, run: Run) -> str:
        """Write the run's records and its summary; returns the new head hash."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        prev = self.last_hash()
        rows: list[dict[str, Any]] = [r.model_dump(mode="json") for r in run.records]
        s = run.summary()
        rows.append({"kind": "run", "run_id": run.run_id, "started_at": run.started_at, "policy_pack": run.pack,
                     "policy_version": run.policy_version, "policy_digest": run.policy_digest,
                     "traces": s["traces"], "records": len(run.records), "findings": s["findings"],
                     "errors": s["errors"], "by_severity": s["by_severity"], "usage": s["usage"], "models": run.models})
        with self.path.open("a") as f:
            for row in rows:
                row["prev_hash"] = prev
                row["hash"] = prev = _hash(prev, row)
                f.write(json.dumps(row, default=str) + "\n")
        return prev

    def verify(self) -> tuple[bool, int, str]:
        """(ok, records checked, message)."""
        if not self.path.exists():
            return True, 0, "no audit log yet"
        prev, n = GENESIS, 0
        with self.path.open() as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    return False, n, f"line {lineno} is not valid JSON"
                if row.get("prev_hash") != prev or _hash(prev, row) != row.get("hash"):
                    return False, n, f"chain broken at line {lineno} ({row.get('kind')} {row.get('record_id') or row.get('run_id')})"
                prev, n = row["hash"], n + 1
        return True, n, f"{n} records, chain intact, head {prev[:12]}"
