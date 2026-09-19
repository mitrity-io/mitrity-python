"""Governed CrewAI tools: admit before ``_run`` / ``_arun``.

A CrewAI ``BaseTool`` executes in-process, and the framework reaches it
through ``to_structured_tool()``, which binds ``_run``. ``govern(tool)``
returns a ``BaseTool`` with the same name, description and schema whose
``_run`` and ``_arun`` ask the MITRITY edge first, return a ``ToolFailure``
carrying the edge's reason on a deny — the framework's declared channel for a
tool that did not do what it was asked, recorded and policy-driven — run the
edge's ``updated_input`` when it sends one, and only then call the tool.

Coverage is exactly what you hand it: the tools CrewAI adds to an agent
itself (delegation, the code interpreter behind ``allow_code_execution``, MCP
tools from ``mcps``) never pass through ``govern`` and are invisible here.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any

import crewai
from crewai.tools import BaseTool
from crewai.tools.base_tool import Tool as DecoratedTool
from crewai.tools.tool_failure import ToolFailure

import mitrity
from mitrity.admission import (
    ADAPTER_NAME,
    AdmissionError,
    AdmitRequest,
    Attestation,
    Client,
    Surface,
    Verdict,
    config_hash,
)

SURFACE: Surface = "crewai"
FRAMEWORK = "crewai"
ADAPTER_VERSION = mitrity.__version__
FRAMEWORK_VERSION: str | None = getattr(crewai, "__version__", None)

_ATTEST_RETRY_SECONDS = 30.0
_PROCESS_SESSION_ID = f"crewai-{uuid.uuid4()}"

logger = logging.getLogger("mitrity.crewai")


class CrewAIGovernor:
    """Shared state for a set of governed tools: the client, the session, the attestation."""

    def __init__(
        self,
        *,
        client: Client | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client if client is not None else Client()
        self._session_id = session_id
        self._cwd = cwd
        self._logger = logger or logging.getLogger("mitrity.crewai")
        self._hooked: list[str] = []
        self._attested: dict[str, str] = {}
        self._attest_attempted: dict[str, float] = {}
        self._lock = threading.Lock()

    @property
    def client(self) -> Client:
        return self._client

    @property
    def hooked_tools(self) -> tuple[str, ...]:
        return tuple(self._hooked)

    def register(self, tool_name: str) -> None:
        with self._lock:
            if tool_name not in self._hooked:
                self._hooked.append(tool_name)

    def attestation(self) -> Attestation:
        hooked = tuple(sorted(self._hooked))
        hashed = {
            "adapter": ADAPTER_NAME,
            "adapter_version": ADAPTER_VERSION,
            "framework": FRAMEWORK,
            "framework_version": FRAMEWORK_VERSION,
            "hooked_tools": list(hooked),
            "unhooked_exec_tools": [],
            "disallowed_tools": [],
            "other_mcp_servers": [],
        }
        return Attestation(
            framework=FRAMEWORK,
            framework_version=FRAMEWORK_VERSION,
            adapter=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            hooked_tools=hooked,
            config_hash=config_hash(hashed),
        )

    def session_id(self) -> str:
        return self._session_id or _PROCESS_SESSION_ID

    def cwd(self) -> str:
        return self._cwd or os.getcwd()

    def _should_attest(self, session_id: str, digest: str) -> bool:
        with self._lock:
            if self._attested.get(session_id) == digest:
                return False
            now = time.monotonic()
            last = self._attest_attempted.get(session_id)
            if (
                last is not None
                and now - last < _ATTEST_RETRY_SECONDS
                and session_id not in self._attested
            ):
                return False
            self._attest_attempted[session_id] = now
            return True

    def ensure_attested(self, session_id: str) -> None:
        attestation = self.attestation()
        digest = attestation.config_hash or ""
        if not self._should_attest(session_id, digest):
            return
        try:
            self._client.attest(attestation)
        except AdmissionError as exc:
            self._logger.warning("MITRITY: could not report the runtime posture: %s", exc)
            return
        with self._lock:
            self._attested[session_id] = digest

    async def ensure_attested_async(self, session_id: str) -> None:
        attestation = self.attestation()
        digest = attestation.config_hash or ""
        if not self._should_attest(session_id, digest):
            return
        try:
            await self._client.attest_async(attestation)
        except AdmissionError as exc:
            self._logger.warning("MITRITY: could not report the runtime posture: %s", exc)
            return
        with self._lock:
            self._attested[session_id] = digest

    def request_for(
        self, tool: BaseTool, args: tuple[Any, ...], kwargs: Mapping[str, Any], *, session_id: str
    ) -> AdmitRequest:
        tool_input: dict[str, Any] = dict(kwargs)
        if args:
            tool_input[_positional_key(tool)] = args[0] if len(args) == 1 else list(args)
        return AdmitRequest(
            surface=SURFACE,
            framework_version=FRAMEWORK_VERSION,
            session_id=session_id,
            cwd=self.cwd(),
            tool_name=tool.name,
            tool_input=tool_input,
        )


class GovernedTool(BaseTool):
    """A ``BaseTool`` that admits every call through the MITRITY edge before running ``inner``."""

    inner: BaseTool
    governor: CrewAIGovernor

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        session_id = self.governor.session_id()
        self.governor.ensure_attested(session_id)
        request = self.governor.request_for(self.inner, args, kwargs, session_id=session_id)
        verdict = self.governor.client.decide(request)
        applied = _apply_verdict(self.inner, verdict, args, kwargs)
        if isinstance(applied, ToolFailure):
            return applied
        args, kwargs = applied
        return self.inner._run(*args, **kwargs)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        session_id = self.governor.session_id()
        await self.governor.ensure_attested_async(session_id)
        request = self.governor.request_for(self.inner, args, kwargs, session_id=session_id)
        verdict = await self.governor.client.decide_async(request)
        applied = _apply_verdict(self.inner, verdict, args, kwargs)
        if isinstance(applied, ToolFailure):
            return applied
        args, kwargs = applied
        if _has_async_implementation(self.inner):
            return await self.inner._arun(*args, **kwargs)
        return await asyncio.to_thread(self.inner._run, *args, **kwargs)


def _has_async_implementation(tool: BaseTool) -> bool:
    """Whether ``_arun`` is the tool's own.

    The framework's fallback ``_arun`` runs the sync function before reporting
    that it is not async, so a sync tool goes to a worker thread instead.
    """
    if isinstance(tool, DecoratedTool):
        return inspect.iscoroutinefunction(tool.func)
    return type(tool)._arun is not BaseTool._arun


def _apply_verdict(
    tool: BaseTool, verdict: Verdict, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]] | ToolFailure:
    if not verdict.allowed:
        return ToolFailure(message=verdict.reason)
    if verdict.updated_input is None:
        return args, dict(kwargs)
    if verdict.routed_to:
        logger.info("MITRITY routed %s to %s", tool.name, verdict.routed_to)
    updated = dict(verdict.updated_input)
    if args:
        # A positional call has exactly one place a rewrite can land; a rewrite
        # the adapter cannot place must not leave the original value to run.
        key = _positional_key(tool)
        if key not in updated or set(updated) != {key}:
            return ToolFailure(
                message=(
                    f"MITRITY rewrote the input of {tool.name} in a way the adapter cannot apply "
                    f"(keys {sorted(updated)!r}); the call was blocked rather than run unchanged"
                )
            )
        args = (updated.pop(key), *args[1:])
    return args, {**kwargs, **updated}


def _positional_key(tool: BaseTool) -> str:
    """The wire key for one positional argument: the single declared argument, else ``input``."""
    try:
        declared = list(tool.args_schema.model_fields)
    except Exception as exc:
        logger.debug(
            "MITRITY: could not introspect the arguments of %s (%r); using 'input'", tool.name, exc
        )
        declared = []
    return declared[0] if len(declared) == 1 else "input"


def govern(
    tool: BaseTool,
    *,
    client: Client | None = None,
    session_id: str | None = None,
    cwd: str | None = None,
    governor: CrewAIGovernor | None = None,
) -> GovernedTool:
    """Wrap one CrewAI tool so every call is admitted by the MITRITY edge first."""
    shared = (
        governor
        if governor is not None
        else CrewAIGovernor(client=client, session_id=session_id, cwd=cwd)
    )
    shared.register(tool.name)
    return GovernedTool(
        inner=tool,
        governor=shared,
        name=tool.name,
        description=tool.description,
        description_updated=tool.description_updated,
        env_vars=list(tool.env_vars),
        args_schema=tool.args_schema,
        result_schema=tool.result_schema,
        cache_function=tool.cache_function,
        result_as_answer=tool.result_as_answer,
        max_usage_count=tool.max_usage_count,
        tool_failure_policy=tool.tool_failure_policy,
    )


def govern_tools(
    tools: Iterable[BaseTool],
    *,
    client: Client | None = None,
    session_id: str | None = None,
    cwd: str | None = None,
) -> list[BaseTool]:
    """Wrap a sequence of tools with one shared governor; order is preserved."""
    shared = CrewAIGovernor(client=client, session_id=session_id, cwd=cwd)
    return [govern(tool, governor=shared) for tool in tools]
