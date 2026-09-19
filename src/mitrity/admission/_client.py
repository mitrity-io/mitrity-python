"""The admission client: one authenticated request, and the two-phase decision.

Every failure — an unreachable socket, a token the edge rejects, a ``503``,
a body that is not a decision, a deadline exceeded — is an exception from
``admit`` and a deny from ``decide``. There is no path through this module
that produces an allow the edge did not send for this request.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import replace
from typing import Any

import httpx

from ._config import (
    ATTEST_TIMEOUT,
    HOLD_MARGIN,
    Config,
    parse_loopback_addr,
    split_addr,
    validate_addr,
)
from ._errors import (
    AdmissionConfigError,
    AdmissionError,
    AdmissionNotReady,
    AdmissionPayloadTooLarge,
    AdmissionProtocolError,
    AdmissionTimeout,
    AdmissionUnauthorized,
    AdmissionUnreachable,
)
from ._types import AdmitRequest, Attestation, Decision, Verdict

PROTOCOL_VERSION = "1"
"""The admission protocol this adapter speaks. Sent on every request; nothing else is accepted."""

HEADER_TOKEN = "X-Mitrity-Admission-Token"
HEADER_VERSION = "X-Mitrity-Admission-Version"

MAX_REQUEST_BYTES = 64 * 1024
"""The edge's request-body cap. A larger input is denied here, without sending it."""

_MAX_RESPONSE_BYTES = 256 * 1024
_HOST = "mitrity-admission"
_SUMMARY_LIMIT = 200

logger = logging.getLogger("mitrity.admission")

UNREACHABLE_HINT = (
    "This is not a policy decision — the governance edge could not be reached. "
    "Check that the MITRITY edge is running and that MITRITY_ADMISSION_ADDR names its "
    "admission socket."
)


def _summarize(body: bytes) -> str:
    """Bound an error body before it reaches a message the model will read."""
    text = body.decode("utf-8", "replace").strip()
    text = "".join(" " if ord(c) < 0x20 or ord(c) == 0x7F else c for c in text)
    if len(text) > _SUMMARY_LIMIT:
        return text[:_SUMMARY_LIMIT] + "…"
    return text


def _read_token(path: str) -> str:
    """Read the per-process admission token, fresh, so a rotated token is picked up."""
    if not path:
        raise AdmissionConfigError(
            "no admission token file configured (set MITRITY_ADMISSION_TOKEN_FILE)"
        )
    try:
        with open(path, encoding="utf-8") as handle:
            token = handle.read().strip()
    except OSError as exc:
        raise AdmissionUnreachable(
            f"admission token file {path!r} could not be read ({exc.strerror or exc}): "
            "the MITRITY edge is not running here, or is not configured to serve admission"
        ) from exc
    if not token:
        raise AdmissionUnreachable(f"admission token file {path!r} is empty")
    return token


def _encode(body: Any) -> bytes | None:
    if body is None:
        return None
    payload = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(payload) > MAX_REQUEST_BYTES:
        raise AdmissionPayloadTooLarge(
            f"the tool input is larger than MITRITY will judge ({len(payload)} bytes, "
            f"cap {MAX_REQUEST_BYTES})"
        )
    return payload


def _interpret(status: int, body: bytes, headers: httpx.Headers, *, addr: str) -> Any:
    """Turn a status and body into parsed JSON, or the error the contract implies."""
    version = headers.get(HEADER_VERSION)
    if version is not None and version != PROTOCOL_VERSION:
        raise AdmissionProtocolError(
            f"admission API at {addr} speaks protocol version {version!r}; this adapter speaks "
            f"{PROTOCOL_VERSION!r} — upgrade the edge or the adapter"
        )
    if status == 204:
        return None
    if status == 200 and version is None:
        # Nothing authenticates the edge to the adapter; the version header is the one
        # signal that the peer is a MITRITY edge. A decision without it is not obeyed.
        raise AdmissionProtocolError(
            f"admission API at {addr} answered without an {HEADER_VERSION} header — "
            "not a MITRITY edge, or a protocol the adapter does not speak"
        )
    if status == 503:
        raise AdmissionNotReady(f"the MITRITY edge is not ready to judge ({_reason_or(body)})")
    if status == 400:
        raise AdmissionProtocolError(
            f"admission API rejected the request (400): {_reason_or(body)}"
        )
    if status != 200:
        raise AdmissionProtocolError(f"admission API returned {status}: {_summarize(body)}")
    try:
        return json.loads(body)
    except ValueError as exc:
        raise AdmissionProtocolError("admission response was not valid JSON") from exc


def _reason_or(body: bytes) -> str:
    """The ``reason``/``error`` field of a JSON error body, else a bounded summary of it."""
    try:
        data = json.loads(body)
    except ValueError:
        return _summarize(body)
    if isinstance(data, dict):
        for key in ("reason", "error"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return _summarize(value.encode("utf-8"))
    return _summarize(body)


def _limits() -> httpx.Limits:
    # No keep-alive: the edge idles connections out and a stale pooled socket
    # would surface as a spurious transport error on the next decision.
    return httpx.Limits(max_keepalive_connections=0)


class Client:
    """Talks to the loopback admission API.

    Safe to share across threads and tasks: it holds configuration only and
    opens one short-lived connection per request, reading the token file each
    time so an edge restart is picked up without the caller caring.
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        addr: str | None = None,
        token_file: str | None = None,
        timeout: float | None = None,
        hold_timeout: float | None = None,
    ) -> None:
        cfg = config if config is not None else Config.from_env()
        cfg = replace(
            cfg,
            addr=addr if addr is not None else cfg.addr,
            token_file=token_file if token_file is not None else cfg.token_file,
            timeout=timeout if timeout is not None else cfg.timeout,
            hold_timeout=hold_timeout if hold_timeout is not None else cfg.hold_timeout,
        )
        self._config = cfg
        self._config_error: AdmissionConfigError | None = None
        try:
            validate_addr(cfg.addr)
        except AdmissionConfigError as exc:
            # Not raised here on purpose: a hook that crashes at construction
            # takes the application down; one that denies every call is what
            # fail-closed means.
            self._config_error = exc
        self._network, self._address = split_addr(cfg.addr)
        self._host = ""
        self._port = 0
        if self._network == "tcp" and self._config_error is None:
            # The URL is built from the parsed literal host and port, never from
            # the configured string, so nothing in it can name another authority.
            self._host, self._port = parse_loopback_addr(self._address)

    @property
    def config(self) -> Config:
        return self._config

    # ------------------------------------------------------------------ sync

    def admit(self, request: AdmitRequest, *, timeout: float | None = None) -> Decision:
        """Ask for a decision. Raises :class:`AdmissionError` on any failure."""
        started = time.monotonic()
        data = self._request(
            "POST", "/v1/admit", request.to_wire(), timeout or self._config.timeout
        )
        decision = Decision.from_wire(data)
        self._log(request, decision, started)
        return decision

    def attest(self, attestation: Attestation, *, timeout: float | None = None) -> None:
        """Report the runtime's posture. Raises on failure; the caller logs, never blocks."""
        self._request("POST", "/v1/attest", attestation.to_wire(), timeout or ATTEST_TIMEOUT)

    def health(self, *, timeout: float | None = None) -> dict[str, Any]:
        """``GET /healthz``, unauthenticated. Useful for a doctor-style check."""
        data = self._request("GET", "/healthz", None, timeout or self._config.timeout, auth=False)
        return data if isinstance(data, dict) else {}

    def decide(self, request: AdmitRequest, *, hold_timeout: float | None = None) -> Verdict:
        """The two-phase decision: never raises, never fabricates an allow.

        Phase 1 asks with ``hold_timeout_seconds: 0`` under the deadline. Only
        a ``held`` answer starts phase 2, which re-submits with the hold budget
        so the edge long-polls the approval. Anything that goes wrong on
        either phase is a deny naming what went wrong. ``hold_timeout`` caps
        the configured budget for this call; it can never raise it.
        """
        try:
            first = self.admit(replace(request, hold_timeout_seconds=0))
        except AdmissionError as exc:
            return _unreachable_verdict(exc)
        if first.decision != "held":
            return _verdict_for(first)
        budget = self._hold_budget(hold_timeout)
        if budget <= 0:
            return _verdict_for(first)
        try:
            second = self.admit(
                replace(request, hold_timeout_seconds=budget), timeout=budget + HOLD_MARGIN
            )
        except AdmissionError as exc:
            return _hold_failed_verdict(first, exc)
        return _verdict_for(second)

    # ----------------------------------------------------------------- async

    async def admit_async(self, request: AdmitRequest, *, timeout: float | None = None) -> Decision:
        started = time.monotonic()
        data = await self._request_async(
            "POST", "/v1/admit", request.to_wire(), timeout or self._config.timeout
        )
        decision = Decision.from_wire(data)
        self._log(request, decision, started)
        return decision

    async def attest_async(self, attestation: Attestation, *, timeout: float | None = None) -> None:
        await self._request_async(
            "POST", "/v1/attest", attestation.to_wire(), timeout or ATTEST_TIMEOUT
        )

    async def health_async(self, *, timeout: float | None = None) -> dict[str, Any]:
        data = await self._request_async(
            "GET", "/healthz", None, timeout or self._config.timeout, auth=False
        )
        return data if isinstance(data, dict) else {}

    async def decide_async(
        self, request: AdmitRequest, *, hold_timeout: float | None = None
    ) -> Verdict:
        try:
            first = await self.admit_async(replace(request, hold_timeout_seconds=0))
        except AdmissionError as exc:
            return _unreachable_verdict(exc)
        if first.decision != "held":
            return _verdict_for(first)
        budget = self._hold_budget(hold_timeout)
        if budget <= 0:
            return _verdict_for(first)
        try:
            second = await self.admit_async(
                replace(request, hold_timeout_seconds=budget), timeout=budget + HOLD_MARGIN
            )
        except AdmissionError as exc:
            return _hold_failed_verdict(first, exc)
        return _verdict_for(second)

    # -------------------------------------------------------------- plumbing

    def _hold_budget(self, cap: float | None) -> int:
        budget = self._config.hold_timeout
        if cap is not None:
            budget = min(budget, max(cap, 0.0))
        return int(budget)

    def _log(self, request: AdmitRequest, decision: Decision, started: float) -> None:
        # The tool input never reaches a log record, and neither does the
        # reason (it names the resolved command). What is logged is enough to
        # line a decision up against the audit trail.
        logger.debug(
            "admission decision tool=%s decision=%s admission_id=%s risk=%.2f routed_to=%s ms=%d",
            request.tool_name,
            decision.decision,
            decision.admission_id,
            decision.risk_score,
            decision.routed_to,
            int((time.monotonic() - started) * 1000),
        )

    def _url(self, path: str) -> str:
        if self._network == "unix":
            return f"http://{_HOST}{path}"
        host = f"[{self._host}]" if ":" in self._host else self._host
        return f"http://{host}:{self._port}{path}"

    def _headers(self, token: str | None, payload: bytes | None) -> dict[str, str]:
        headers = {"Host": _HOST, HEADER_VERSION: PROTOCOL_VERSION}
        if token is not None:
            headers[HEADER_TOKEN] = token
        if payload is not None:
            headers["Content-Type"] = "application/json"
        return headers

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AdmissionTimeout("deadline exceeded before the request was sent")
        return remaining

    def _timeout_error(self, timeout: float) -> AdmissionTimeout:
        return AdmissionTimeout(
            f"admission API at {self._config.addr} did not answer within {timeout * 1000:.0f} ms"
        )

    def _unreachable_error(self, exc: Exception) -> AdmissionUnreachable:
        return AdmissionUnreachable(f"admission API at {self._config.addr} unreachable: {exc}")

    def _request(
        self, method: str, path: str, body: Any, timeout: float, *, auth: bool = True
    ) -> Any:
        if self._config_error is not None:
            raise self._config_error
        deadline = time.monotonic() + timeout
        payload = _encode(body)
        attempt = 0
        while True:
            attempt += 1
            remaining = self._remaining(deadline)
            token = _read_token(self._config.token_file) if auth else None
            transport = (
                httpx.HTTPTransport(uds=self._address, retries=0)
                if self._network == "unix"
                else httpx.HTTPTransport(retries=0)
            )
            try:
                with (
                    httpx.Client(
                        transport=transport, timeout=httpx.Timeout(remaining), limits=_limits()
                    ) as http,
                    http.stream(
                        method,
                        self._url(path),
                        content=payload,
                        headers=self._headers(token, payload),
                    ) as response,
                ):
                    status = response.status_code
                    headers = response.headers
                    data = _collect(response.iter_bytes())
            except httpx.TimeoutException as exc:
                raise self._timeout_error(timeout) from exc
            except httpx.HTTPError as exc:
                raise self._unreachable_error(exc) from exc
            if status == 401:
                # Exactly one retry with a freshly-read token: the edge may have
                # restarted and minted a new one. A file that keeps producing
                # 401s is a broken deployment, not something to loop on.
                if attempt == 1 and deadline - time.monotonic() > 0:
                    continue
                raise AdmissionUnauthorized(
                    "admission token rejected (401) after re-reading the token file"
                )
            return _interpret(status, data, headers, addr=self._config.addr)

    async def _request_async(
        self, method: str, path: str, body: Any, timeout: float, *, auth: bool = True
    ) -> Any:
        if self._config_error is not None:
            raise self._config_error
        deadline = time.monotonic() + timeout
        payload = _encode(body)
        attempt = 0
        while True:
            attempt += 1
            remaining = self._remaining(deadline)
            token = _read_token(self._config.token_file) if auth else None
            transport = (
                httpx.AsyncHTTPTransport(uds=self._address, retries=0)
                if self._network == "unix"
                else httpx.AsyncHTTPTransport(retries=0)
            )
            try:
                async with (
                    httpx.AsyncClient(
                        transport=transport, timeout=httpx.Timeout(remaining), limits=_limits()
                    ) as http,
                    http.stream(
                        method,
                        self._url(path),
                        content=payload,
                        headers=self._headers(token, payload),
                    ) as response,
                ):
                    status = response.status_code
                    headers = response.headers
                    data = await _collect_async(response.aiter_bytes())
            except httpx.TimeoutException as exc:
                raise self._timeout_error(timeout) from exc
            except httpx.HTTPError as exc:
                raise self._unreachable_error(exc) from exc
            if status == 401:
                if attempt == 1 and deadline - time.monotonic() > 0:
                    continue
                raise AdmissionUnauthorized(
                    "admission token rejected (401) after re-reading the token file"
                )
            return _interpret(status, data, headers, addr=self._config.addr)


def _collect(chunks: Iterator[bytes]) -> bytes:
    """Read a response body, bounded: a decision is a few hundred bytes."""
    data = bytearray()
    for chunk in chunks:
        data.extend(chunk)
        if len(data) > _MAX_RESPONSE_BYTES:
            raise AdmissionProtocolError(
                "admission response exceeded the size an adapter will read"
            )
    return bytes(data)


async def _collect_async(chunks: AsyncIterator[bytes]) -> bytes:
    data = bytearray()
    async for chunk in chunks:
        data.extend(chunk)
        if len(data) > _MAX_RESPONSE_BYTES:
            raise AdmissionProtocolError(
                "admission response exceeded the size an adapter will read"
            )
    return bytes(data)


def _verdict_for(decision: Decision) -> Verdict:
    if decision.decision == "allow":
        return Verdict(
            allowed=True,
            reason=decision.reason,
            decision=decision,
            updated_input=decision.updated_input,
            routed_to=decision.routed_to,
        )
    if decision.decision == "held":
        return Verdict(
            allowed=False,
            reason=(
                "MITRITY is holding this action for human approval "
                f"(approval {decision.approval_id or 'unknown'}) and it has not been approved. "
                "Ask the operator to approve it in the MITRITY console, then try again."
            ),
            decision=decision,
            held=True,
        )
    reason = decision.reason or "MITRITY policy denied this action"
    return Verdict(allowed=False, reason=f"MITRITY denied this action: {reason}", decision=decision)


def _unreachable_verdict(exc: AdmissionError) -> Verdict:
    return Verdict(
        allowed=False,
        reason=f"MITRITY could not authorize this action and blocked it: {exc}. {UNREACHABLE_HINT}",
        error=exc,
    )


def _hold_failed_verdict(held: Decision, exc: AdmissionError) -> Verdict:
    return Verdict(
        allowed=False,
        reason=(
            "MITRITY held this action for human approval "
            f"(approval {held.approval_id or 'unknown'}) and waiting on the approval "
            f"failed: {exc}. "
            "The action has not been approved."
        ),
        decision=held,
        error=exc,
        held=True,
    )
