"""The Jev client (TypeSafe direct).

    jev = Jev()                                             # TYPESAFE_API_KEY from the environment
    r = await jev.ask({"claim": "...", "evidence": "..."},
                      {"supported": YesNo("Is claim supported by evidence?")})
    r.answers["supported"].p, r.usage

Anything with the same `ask()` signature is a backend (see `Backend`); `FakeJev` is the test one.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import httpx

from .types import JevResponse, Question, Usage
from .wire import decode_answers, encode_request

DEFAULT_BASE_URL = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"  # required by the API; alias for the current Jev version
PRICE_PER_M_INPUT_USD = 0.042  # output tokens are not billed


@runtime_checkable
class Backend(Protocol):
    name: str

    async def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse: ...


class JevError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status == 429 or self.status >= 500


class Jev:
    name = "jev"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 15.0,  # Jev usually answers in well under a second; fail fast and retry
        max_retries: int = 3,
        zero_data_retention: bool = True,
        http: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key:
            raise JevError("TYPESAFE_API_KEY is not set and no api_key was passed")
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("TYPESAFE_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries
        self.zero_data_retention = zero_data_retention
        self._own_http = http is None
        self._http = http
        self._loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ http

    def _client(self) -> httpx.AsyncClient:
        # An AsyncClient is tied to the event loop it was first used on. sync callers (evaluate_sync) run a fresh loop
        # each call, so when we own the client, rebuild it whenever the loop changes.
        if not self._own_http:
            return self._http  # type: ignore[return-value]
        loop = asyncio.get_running_loop()
        if self._http is None or self._http.is_closed or self._loop is not loop:
            self._http = httpx.AsyncClient(timeout=self.timeout)
            self._loop = loop
        return self._http

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
             "User-Agent": "evaltrix"}
        if self.zero_data_retention:
            h["X-Data-Retention"] = "zero"
        return h

    async def _post_once(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            r = await self._client().post(f"{self.base_url}/v1/systemone", json=body, headers=self._headers())
        except httpx.TransportError as e:
            raise JevError(f"jev: {type(e).__name__}: {e}") from e
        if r.status_code >= 400:
            ra = r.headers.get("retry-after")
            raise JevError(f"jev {r.status_code}: {r.text[:300]}", status=r.status_code,
                           retry_after=float(ra) if ra and ra.replace(".", "", 1).isdigit() else None)
        try:
            return r.json()
        except ValueError as e:
            raise JevError(f"jev: response is not JSON: {r.text[:200]}", status=r.status_code) from e

    # ------------------------------------------------------------------ public

    async def ask(self, state: Any, questions: Mapping[str, Question]) -> JevResponse:
        if not questions:
            return JevResponse(answers={})
        body = encode_request(state, questions, self.model)
        t0 = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                data = await self._post_once(body)
                break
            except JevError as e:
                if not e.retryable or attempt == self.max_retries:
                    raise
                delay = 0.5 * 2**attempt + random.uniform(0, 0.25)
                if e.retry_after:
                    delay = max(delay, min(e.retry_after, 30.0))
                await asyncio.sleep(delay)
        u = data.get("usage") or {}
        input_tokens = int(u.get("input_tokens") or 0)
        usage = Usage(
            requests=1,
            input_tokens=input_tokens,
            output_tokens=int(u.get("output_tokens") or 0),
            latency_ms=(time.perf_counter() - t0) * 1000,
            cost_usd=input_tokens / 1e6 * PRICE_PER_M_INPUT_USD,
        )
        return JevResponse(answers=decode_answers(data.get("answers") or {}, questions), model=data.get("model"),
                           usage=usage, raw=data)

    async def aclose(self) -> None:
        if self._own_http and self._http is not None:
            await self._http.aclose()

    async def __aenter__(self) -> Jev:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()


def run_sync(coro: Any) -> Any:
    """Run a coroutine from sync code, including when a loop is already running (Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()
