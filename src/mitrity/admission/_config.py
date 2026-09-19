"""Discovery of the edge: the hook's environment variables, defaults and rules."""

from __future__ import annotations

import ipaddress
import os
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ._errors import AdmissionConfigError

ENV_ADDR = "MITRITY_ADMISSION_ADDR"
ENV_TOKEN_FILE = "MITRITY_ADMISSION_TOKEN_FILE"
ENV_TIMEOUT = "MITRITY_HOOK_TIMEOUT"
ENV_HOLD_TIMEOUT = "MITRITY_HOOK_HOLD_TIMEOUT"
# Read for documentation's sake and deliberately ignored: adapters have no
# fail-open mode (adapters.md, G2).
ENV_FAIL_MODE = "MITRITY_HOOK_FAIL_MODE"

DEFAULT_TIMEOUT = 0.5
"""Deadline for one decision, in seconds. The cached-policy path is sub-millisecond."""

MAX_TIMEOUT = 30.0

DEFAULT_HOLD_TIMEOUT = 540.0
"""How long to wait on a human approval, in seconds. ``0`` disables waiting."""

MAX_HOLD_TIMEOUT = 570.0
"""Ceiling sized against the framework's own 600 s hook budget, as for the hook."""

HOLD_MARGIN = 5.0
"""Headroom over the hold budget so the edge's own long-poll can answer first."""

ATTEST_TIMEOUT = 5.0
"""Deadline for ``POST /v1/attest``. Nothing is blocked on it, so it may be longer."""

Network = Literal["unix", "tcp"]

_DURATION_UNITS = {
    "ns": 1e-9,
    "us": 1e-6,
    "µs": 1e-6,
    "μs": 1e-6,
    "ms": 1e-3,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
}
_DURATION_PART = re.compile(r"(\d+(?:\.\d*)?|\.\d+)(ns|us|µs|μs|ms|s|m|h)")
_BARE_NUMBER = re.compile(r"^[+-]?(\d+(?:\.\d*)?|\.\d+)$")
# A loopback address is exactly host:port — no userinfo, path, query, fragment or
# whitespace, so nothing an HTTP client could read as a different authority.
_TCP_ADDR = re.compile(
    r"^(?:\[(?P<bracket>[0-9A-Fa-f:.]+)\]|(?P<bare>[0-9A-Za-z.\-]+)):(?P<port>\d{1,5})$"
)


def platform_defaults() -> tuple[str, str]:
    """The admission address and token path a standard install uses on this platform.

    A Unix socket wherever one can be dialed; loopback TCP on Windows, where the
    edge listens on a port because Go cannot dial a named pipe.
    """
    if sys.platform == "win32":
        program_data = os.environ.get("PROGRAMDATA") or r"C:\ProgramData"
        return "127.0.0.1:8777", os.path.join(program_data, "Mitrity", "admission.token")
    return "unix:/run/mitrity/admission.sock", "/run/mitrity/admission.token"


def parse_duration(value: str | None, fallback: float) -> float:
    """Parse a Go duration (``500ms``, ``9m``, ``1h30m``) or a bare number of seconds.

    Malformed input falls back rather than failing: a misconfigured timeout must
    not become a crashed hook and an ungoverned session. Mirrors the hook's
    ``durationOr``.
    """
    if value is None:
        return fallback
    text = value.strip()
    if not text:
        return fallback
    if _BARE_NUMBER.match(text):
        return float(text)
    sign = 1.0
    if text[0] in "+-":
        sign = -1.0 if text[0] == "-" else 1.0
        text = text[1:]
    total = 0.0
    position = 0
    for match in _DURATION_PART.finditer(text):
        if match.start() != position:
            return fallback
        total += float(match.group(1)) * _DURATION_UNITS[match.group(2)]
        position = match.end()
    if position != len(text) or position == 0:
        return fallback
    return sign * total


def split_addr(addr: str) -> tuple[Network, str]:
    """Split a configured address into the network and the dial address, as the edge does."""
    if addr.startswith("unix://"):
        return "unix", addr[len("unix://") :]
    if addr.startswith("unix:"):
        return "unix", addr[len("unix:") :]
    if addr.startswith("/"):
        return "unix", addr
    return "tcp", addr


def parse_loopback_addr(address: str) -> tuple[str, int]:
    """Parse a loopback ``host:port`` strictly, returning the literal host and the port.

    The address is parsed, never sliced: ``localhost:8777@attacker.example`` has a
    loopback-looking prefix and an HTTP client would connect to the host after the
    ``@``. Only ``host:port`` is accepted, the host must be a loopback IP literal —
    ``localhost`` is taken as a spelling of ``127.0.0.1`` and never resolved, so a
    hosts-file entry cannot point it off-box — and the port must be a number.
    """
    match = _TCP_ADDR.match(address)
    if match is None:
        raise AdmissionConfigError(
            f"{ENV_ADDR}={address!r} is not a plain host:port: the admission address may carry "
            "no userinfo, path, query, fragment or whitespace"
        )
    host = match.group("bracket") or match.group("bare")
    port = int(match.group("port"))
    if not 1 <= port <= 65535:
        raise AdmissionConfigError(f"{ENV_ADDR}={address!r} has an invalid port")
    if host == "localhost":
        host = "127.0.0.1"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError as exc:
        raise AdmissionConfigError(
            f"{ENV_ADDR}={address!r} is neither loopback nor a Unix socket: the admission API "
            "carries the command this agent is about to run and its token, and a routable "
            "address would send both to whatever answers there"
        ) from exc
    if not ip.is_loopback:
        raise AdmissionConfigError(
            f"{ENV_ADDR}={address!r} is neither loopback nor a Unix socket: the admission API "
            "carries the command this agent is about to run and its token, and a routable "
            "address would send both to whatever answers there"
        )
    return str(ip), port


def validate_addr(addr: str) -> None:
    """Refuse an address that is neither loopback nor a Unix socket.

    The admission API carries the command the agent is about to run and the
    token that authorizes judging it. A routable address would send both, in
    cleartext, to whatever answers there — and then act on its answer. The
    check is the hook's, applied before any I/O.
    """
    network, address = split_addr(addr)
    if network == "unix":
        if not address:
            raise AdmissionConfigError(f"{ENV_ADDR}={addr!r} names no socket path")
        return
    parse_loopback_addr(address)


@dataclass(frozen=True)
class Config:
    """Where the edge is and how long to wait for it.

    Explicit values win over the environment; the environment wins over the
    platform defaults. Timeouts are clamped to the same ceilings the hook
    applies, for the same reason: past the framework's own hook budget a
    patient adapter is an irrelevant one.
    """

    addr: str
    token_file: str
    timeout: float = DEFAULT_TIMEOUT
    hold_timeout: float = DEFAULT_HOLD_TIMEOUT

    def __post_init__(self) -> None:
        timeout = self.timeout if self.timeout > 0 else DEFAULT_TIMEOUT
        timeout = min(timeout, MAX_TIMEOUT)
        hold = max(self.hold_timeout, 0.0)
        hold = min(hold, MAX_HOLD_TIMEOUT)
        object.__setattr__(self, "timeout", timeout)
        object.__setattr__(self, "hold_timeout", hold)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> Config:
        """Build a configuration from the hook's environment variables."""
        env = os.environ if environ is None else environ
        default_addr, default_token = platform_defaults()
        return cls(
            addr=(env.get(ENV_ADDR) or "").strip() or default_addr,
            token_file=(env.get(ENV_TOKEN_FILE) or "").strip() or default_token,
            timeout=parse_duration(env.get(ENV_TIMEOUT), DEFAULT_TIMEOUT),
            hold_timeout=parse_duration(env.get(ENV_HOLD_TIMEOUT), DEFAULT_HOLD_TIMEOUT),
        )

    @property
    def network(self) -> Network:
        return split_addr(self.addr)[0]

    @property
    def address(self) -> str:
        return split_addr(self.addr)[1]
