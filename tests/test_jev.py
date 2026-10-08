from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import pytest

from evaltrix.jev import (
    Choice,
    ChoiceAnswer,
    FakeJev,
    Jev,
    JevError,
    ProtocolError,
    Score,
    ScoreAnswer,
    YesNo,
    YesNoAnswer,
)
from evaltrix.jev.wire import decode_answers, encode_request

QUESTIONS = {
    "supported": YesNo("Is claim supported by evidence?", true="Restates evidence", false="Adds facts"),
    "action": Choice("Should this call run?", options={"approve": "Read-only", "escalate": "Moves money", "block": "Off-goal"}),
    "relevance": Score("How well does answer address question?", levels=["not at all", "partly", "fully"]),
}
OK_BODY = {
    "answers": {
        "supported": {"noul": 0.83},
        "action": {"choice": "escalate", "probabilities": {"approve": 0.1, "escalate": 0.7, "block": 0.2}},
        "relevance": {"score": 1.8, "probabilities": {"0": 0.05, "1": 0.1, "2": 0.85}},
    },
    "usage": {"input_tokens": 1000, "output_tokens": 12},
}


# --------------------------------------------------------------------------- types


def test_question_validation():
    with pytest.raises(ValueError):
        Choice("x", options={"only": "one"})
    with pytest.raises(ValueError):
        Score("x", levels=["one"])
    with pytest.raises(ValueError):
        YesNo("x", true="yes only")


def test_answers_normalize_and_derive():
    c = ChoiceAnswer(probs={"a": 2.0, "b": 2.0})
    assert c.probs == {"a": 0.5, "b": 0.5} and c.confidence == 0.0
    s = ScoreAnswer(probs=[0, 0, 1])
    assert s.expected == 2 and s.normalized == 1.0 and s.top == 2 and s.confidence == 1.0
    assert YesNoAnswer(p=0.7).yes


# --------------------------------------------------------------------------- wire


def test_encode_request():
    body = encode_request({"claim": "c"}, QUESTIONS, model="jev-1")
    assert body["model"] == "jev-1"
    assert body["questions"]["supported"] == {
        "type": "noul", "instructions": "Is claim supported by evidence?",
        "criteria": {"true": "Restates evidence", "false": "Adds facts"},
    }
    assert body["questions"]["action"]["criteria"]["escalate"] == "Moves money"
    assert body["questions"]["relevance"]["criteria"] == ["not at all", "partly", "fully"]
    assert "criteria" not in encode_request({}, {"q": YesNo("plain?")})["questions"]["q"]


def test_decode_answers():
    a = decode_answers(OK_BODY["answers"], QUESTIONS)
    assert a["supported"].p == 0.83
    assert a["action"].top == "escalate" and a["action"].p("block") == pytest.approx(0.2)
    assert a["relevance"].top == 2


@pytest.mark.parametrize(
    "answers, match",
    [
        ({}, "no answer"),
        ({**OK_BODY["answers"], "supported": {"oops": 1}}, "no probability"),
        ({**OK_BODY["answers"], "action": {"probabilities": {"approve": 1, "maybe": 0}}}, "didn't ask"),
        ({**OK_BODY["answers"], "relevance": {"probabilities": {"3": 1}}}, "out of range"),
    ],
)
def test_decode_rejects_mismatched_answers(answers, match):
    with pytest.raises(ProtocolError, match=match):
        decode_answers(answers, QUESTIONS)


FIXTURE = Path(__file__).parent / "fixtures" / "typesafe_response.json"


def test_decodes_real_jev_response():
    """A response captured from the live API, so the decoder stays honest without spending credit."""
    raw = json.loads(FIXTURE.read_text())
    qs = {"supported": YesNo("?"), "bad_supported": YesNo("?"), **{k: QUESTIONS[k] for k in ("action", "relevance")}}
    a = decode_answers(raw["answers"], qs)
    assert a["supported"].p == 0.98 and a["bad_supported"].p == 0.03
    assert a["action"].top == "approve" and a["action"].confidence == 0.72  # Jev's own, not recomputed
    assert a["relevance"].top == 2 and a["relevance"].confidence == 0.0


# --------------------------------------------------------------------------- client


def _jev(handler, **kw) -> Jev:
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Jev(api_key="test-key", http=http, **kw)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    async def instant(_):
        return None

    monkeypatch.setattr("evaltrix.jev.client.asyncio.sleep", instant)


async def test_ask_sends_request_and_parses_response():
    seen = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"], seen["body"] = str(req.url), req.headers, json.loads(req.content)
        return httpx.Response(200, json=OK_BODY)

    r = await _jev(handler).ask({"claim": "c"}, QUESTIONS)
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["headers"]["authorization"] == "Bearer test-key"
    assert seen["headers"]["x-data-retention"] == "zero"
    assert seen["body"]["state"] == {"claim": "c"}
    assert seen["body"]["model"] == "jev-latest"  # the API rejects requests without it (422)
    assert r.answers["supported"].p == 0.83
    assert r.model is None  # OK_BODY has no model field
    assert r.usage.requests == 1 and r.usage.input_tokens == 1000
    assert r.usage.cost_usd == pytest.approx(1000 / 1e6 * 0.042)


async def test_retries_429_and_5xx_then_succeeds():
    codes = iter([429, 503])

    def handler(req):
        code = next(codes, 200)
        return httpx.Response(code, json=OK_BODY if code == 200 else {"error": "busy"})

    r = await _jev(handler).ask({}, QUESTIONS)
    assert r.answers["action"].top == "escalate"


async def test_retries_transport_errors():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx.ConnectError("reset")
        return httpx.Response(200, json=OK_BODY)

    await _jev(handler).ask({}, QUESTIONS)
    assert calls["n"] == 2


async def test_does_not_retry_client_errors():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(400, json={"error": "bad question"})

    with pytest.raises(JevError, match="400") as e:
        await _jev(handler).ask({}, QUESTIONS)
    assert e.value.status == 400 and calls["n"] == 1


async def test_gives_up_after_max_retries():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(500)

    with pytest.raises(JevError, match="500"):
        await _jev(handler, max_retries=2).ask({}, QUESTIONS)
    assert calls["n"] == 3


async def test_empty_questions_make_no_request():
    def handler(req):
        raise AssertionError("should not be called")

    r = await _jev(handler).ask({}, {})
    assert r.answers == {} and r.usage.requests == 0


def test_missing_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(JevError, match="TYPESAFE_API_KEY"):
        Jev()


# --------------------------------------------------------------------------- fake


async def test_fake_fixed_glob_rules_and_defaults():
    fake = FakeJev(
        fixed={"supported": True, "act*": "block"},
        rules=lambda qid, q, state: 2 if qid == "relevance" and state.get("good") else None,
    )
    r = await fake.ask({"good": True}, QUESTIONS)
    assert r.answers["supported"].p == 0.95
    assert r.answers["action"].top == "block"
    assert r.answers["relevance"].top == 2
    r2 = await fake.ask({}, QUESTIONS)
    assert r2.answers["relevance"].confidence == 0.0  # uniform fallback
    assert len(fake.calls) == 2


# --------------------------------------------------------------------------- live


@pytest.mark.live
@pytest.mark.skipif(not os.environ.get("TYPESAFE_API_KEY"), reason="TYPESAFE_API_KEY not set")
async def test_live_jev_roundtrip(tmp_path):
    async with Jev() as jev:
        r = await jev.ask(
            {"evidence": "Order A123 shipped Monday, arriving Tuesday.", "claim": "Order A123 arrives Tuesday.",
             "bad_claim": "Order A123 ships free overnight."},
            {
                "supported": YesNo("Is claim supported by evidence?"),
                "bad_supported": YesNo("Is bad_claim supported by evidence?"),
                **{k: QUESTIONS[k] for k in ("action", "relevance")},
            },
        )
    if os.environ.get("EVALTRIX_SAVE_FIXTURE"):  # refresh the offline fixture deliberately, not on every run
        FIXTURE.parent.mkdir(exist_ok=True)
        FIXTURE.write_text(json.dumps(r.raw, indent=2))
    assert r.model and r.model.startswith("jev")
    assert r.answers["supported"].p > r.answers["bad_supported"].p
    assert r.usage.input_tokens > 0
