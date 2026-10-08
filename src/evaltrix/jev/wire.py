"""Translate between evaltrix types and the TypeSafe System One wire format.

Request:   {"state": ..., "questions": {id: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}}}
Response:  {"model": "jev-1.13.0", "answers": {id: ...}, "usage": {"input_tokens", "output_tokens"}}
           (verified live 2026-10-03; a real response is kept in tests/fixtures/typesafe_response.json)

Answer shapes accepted (documented shape first, tolerant fallbacks after):
  noul    {"noul": p}                    | {"probability": p} | {"probabilities": {"true": p}}
  choice  {"choice": k, "probabilities": {option: p}, "confidence": c}
  score   {"score": x, "probabilities": {"0": p, ...}, "confidence": c, "legend": {"0": level, ...}}
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .types import Answer, Choice, ChoiceAnswer, Question, Score, ScoreAnswer, YesNo, YesNoAnswer


class ProtocolError(ValueError):
    """Jev's response didn't match the questions we sent."""


def encode_question(q: Question) -> dict[str, Any]:
    if isinstance(q, YesNo):
        d: dict[str, Any] = {"type": "noul", "instructions": q.instructions}
        if q.true is not None:
            d["criteria"] = {"true": q.true, "false": q.false}
        return d
    if isinstance(q, Choice):
        return {"type": "choice", "instructions": q.instructions, "criteria": dict(q.options)}
    if isinstance(q, Score):
        return {"type": "score", "instructions": q.instructions, "criteria": list(q.levels)}
    raise TypeError(f"not a question: {q!r}")


def encode_request(state: Any, questions: Mapping[str, Question], model: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"state": state, "questions": {qid: encode_question(q) for qid, q in questions.items()}}
    if model:
        body["model"] = model
    return body


def _prob(v: Any, where: str) -> float:
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        raise ProtocolError(f"{where}: expected a probability, got {v!r}") from None


def _confidence(raw: Mapping[str, Any], qid: str) -> float | None:
    return _prob(raw["confidence"], qid) if raw.get("confidence") is not None else None


def decode_answer(raw: Mapping[str, Any], q: Question, qid: str) -> Answer:
    if isinstance(q, YesNo):
        for key in ("noul", "probability", "p"):
            if key in raw:
                return YesNoAnswer(p=_prob(raw[key], qid))
        probs = raw.get("probabilities")
        if isinstance(probs, Mapping) and "true" in probs:
            return YesNoAnswer(p=_prob(probs["true"], qid))
        raise ProtocolError(f"{qid}: no probability in yes/no answer {dict(raw)!r}")

    probs = raw.get("probabilities")
    if not isinstance(probs, Mapping):
        raise ProtocolError(f"{qid}: missing probabilities in {dict(raw)!r}")

    if isinstance(q, Choice):
        unknown = set(map(str, probs)) - set(q.options)
        if unknown:
            raise ProtocolError(f"{qid}: answer has options we didn't ask for: {sorted(unknown)}")
        return ChoiceAnswer(probs={k: _prob(probs.get(k, 0.0), qid) for k in q.options},
                            jev_confidence=_confidence(raw, qid))

    if isinstance(q, Score):
        n = len(q.levels)
        by_idx = {int(k): _prob(v, qid) for k, v in probs.items()}
        if any(i < 0 or i >= n for i in by_idx):
            raise ProtocolError(f"{qid}: level index out of range 0..{n - 1}: {sorted(by_idx)}")
        return ScoreAnswer(probs=[by_idx.get(i, 0.0) for i in range(n)], jev_confidence=_confidence(raw, qid))

    raise TypeError(f"not a question: {q!r}")


def decode_answers(raw: Mapping[str, Any], questions: Mapping[str, Question]) -> dict[str, Answer]:
    missing = [qid for qid in questions if qid not in raw]
    if missing:
        raise ProtocolError(f"Jev returned no answer for {missing}")
    return {qid: decode_answer(raw[qid], q, qid) for qid, q in questions.items()}
