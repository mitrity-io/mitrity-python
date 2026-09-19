"""An in-process fake of the admission API, served over a Unix socket.

It speaks exactly the wire shape of iag-specs ``sentinel/admission-api.md``
and nothing more: token header, version header, ``/v1/admit``, ``/v1/attest``,
``/healthz``. Responses are scripted per test; every request is recorded so a
test can assert what the adapter sent — and, just as often, what it did not.
"""

from __future__ import annotations

import json
import os
import shutil
import socketserver
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

HEADER_TOKEN = "X-Mitrity-Admission-Token"
HEADER_VERSION = "X-Mitrity-Admission-Version"


@dataclass
class Recorded:
    method: str
    path: str
    headers: dict[str, str]
    body: Any
    raw: bytes


@dataclass
class Scripted:
    """One scripted response. ``body`` may be a dict (JSON), bytes (verbatim) or ``None``."""

    status: int = 200
    body: Any = None
    delay: float = 0.0
    version: str | None = "1"
    content_type: str | None = "application/json"


Responder = Callable[[Recorded], Scripted]


def allow(**extra: Any) -> Scripted:
    return Scripted(
        body={
            "decision": "allow",
            "reason": "allowed",
            "approval_id": None,
            "risk_score": 0.1,
            "admission_id": "adm-allow",
            "updated_input": None,
            **extra,
        }
    )


def deny(
    reason: str = 'policy rule "no destructive commands" denied resolved command: rm',
) -> Scripted:
    return Scripted(
        body={
            "decision": "deny",
            "reason": reason,
            "approval_id": None,
            "risk_score": 0.9,
            "admission_id": "adm-deny",
            "updated_input": None,
        }
    )


def held(approval_id: str = "apr-1") -> Scripted:
    return Scripted(
        body={
            "decision": "held",
            "reason": "policy requires human approval",
            "approval_id": approval_id,
            "risk_score": 0.5,
            "admission_id": "adm-held",
            "updated_input": None,
        }
    )


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    # A real edge listens with the kernel's backlog; socketserver's default of 5
    # refuses connections under a burst of concurrent hook calls.
    request_queue_size = 128


class FakeEdge:
    """Start with ``FakeEdge(tmp_path)``; use ``edge.addr`` / ``edge.token_file`` for the client."""

    def __init__(self, root: Path) -> None:
        self.root = root
        # AF_UNIX paths are capped at 104 bytes on macOS (108 on Linux) and a
        # pytest tmp_path is longer than that; the socket lives in a short
        # directory of its own, the token file where the test can see it.
        self._socket_dir = Path(tempfile.mkdtemp(prefix="me-", dir="/tmp"))
        self.socket_path = self._socket_dir / "admission.sock"
        self.token_file = root / "admission.token"
        self.token = "tok-" + os.urandom(12).hex()
        self._write_token(self.token)
        self.requests: list[Recorded] = []
        self._script: list[Scripted | Responder] = []
        self._lock = threading.Lock()
        self.default_admit: Scripted | Responder = allow()
        self.default_attest = Scripted(status=204, body=None, content_type=None)
        self.default_health = Scripted(body={"status": "ok", "profile_age_seconds": 3})
        # Test switches: rotate the token as an edge restart would, right before
        # checking the next request (so the presented token is stale), or reject
        # every token regardless of what is presented.
        self.rotate_on_next_request = False
        self.reject_all_tokens = False
        edge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _serve(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = json.loads(raw) if raw else None
                except ValueError:
                    body = raw
                recorded = Recorded(
                    method=self.command,
                    path=self.path,
                    headers={k: v for k, v in self.headers.items()},
                    body=body,
                    raw=raw,
                )
                with edge._lock:
                    edge.requests.append(recorded)
                response = edge._respond(recorded)
                if response.delay:
                    time.sleep(response.delay)
                payload = b""
                if isinstance(response.body, bytes):
                    payload = response.body
                elif response.body is not None:
                    payload = json.dumps(response.body).encode("utf-8")
                self.send_response(response.status)
                if response.version is not None:
                    self.send_header(HEADER_VERSION, response.version)
                if response.content_type and payload:
                    self.send_header("Content-Type", response.content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                if payload:
                    self.wfile.write(payload)

            def do_POST(self) -> None:
                self._serve()

            def do_GET(self) -> None:
                self._serve()

        self._server = _ThreadingUnixServer(str(self.socket_path), Handler)
        os.chmod(self.socket_path, 0o600)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------- scripting

    @property
    def addr(self) -> str:
        return f"unix:{self.socket_path}"

    def script(self, *responses: Scripted | Responder) -> None:
        """Queue responses for the next ``/v1/admit`` calls, in order."""
        with self._lock:
            self._script.extend(responses)

    def rotate_token(self) -> str:
        """Mint a new token, as an edge restart does. The old one is rejected from now on."""
        self.token = "tok-" + os.urandom(12).hex()
        self._write_token(self.token)
        return self.token

    def admits(self) -> list[Recorded]:
        return [r for r in self.requests if r.path == "/v1/admit"]

    def attests(self) -> list[Recorded]:
        return [r for r in self.requests if r.path == "/v1/attest"]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        shutil.rmtree(self._socket_dir, ignore_errors=True)

    # -------------------------------------------------------------- internals

    def _write_token(self, token: str) -> None:
        self.token_file.write_text(token + "\n")
        os.chmod(self.token_file, 0o600)

    def _respond(self, request: Recorded) -> Scripted:
        if request.path == "/healthz":
            return self.default_health
        version = request.headers.get(HEADER_VERSION)
        if self.rotate_on_next_request:
            self.rotate_on_next_request = False
            self.rotate_token()
        if self.reject_all_tokens or request.headers.get(HEADER_TOKEN) != self.token:
            return Scripted(status=401, body=None, content_type=None)
        if version != "1":
            return Scripted(
                status=400, body={"error": f'{HEADER_VERSION} must be "1", got {version!r}'}
            )
        if request.path == "/v1/attest":
            return self.default_attest
        if request.path == "/v1/admit":
            with self._lock:
                scripted = self._script.pop(0) if self._script else self.default_admit
            return scripted(request) if callable(scripted) else scripted
        return Scripted(status=404, body={"error": "not found"})


@dataclass
class HoldScript:
    """Answers ``held`` to a no-wait request and ``allow``/``deny`` to the waiting one."""

    outcome: Scripted = field(default_factory=allow)
    wait: float = 0.05
    approval_id: str = "apr-hold"

    def __call__(self, request: Recorded) -> Scripted:
        budget = (
            request.body.get("hold_timeout_seconds") if isinstance(request.body, dict) else None
        )
        if budget == 0:
            return held(self.approval_id)
        return Scripted(
            status=self.outcome.status,
            body=self.outcome.body,
            delay=self.wait,
            version=self.outcome.version,
        )
