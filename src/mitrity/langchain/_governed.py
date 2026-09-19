"""Governed LangChain tools: admit before ``_run`` / ``_arun``.

A LangChain tool executes in-process, so there is no hook to install: the
adapter wraps the tool. ``govern(tool)`` returns a ``BaseTool`` with the same
name, description and schema whose ``_run`` and ``_arun`` ask the MITRITY edge
first, raise ``ToolException`` with the edge's reason on a deny, run the
edge's ``updated_input`` when it sends one, and only then call the tool.

Coverage is exactly what you hand it: a tool that was not passed through
``govern`` is invisible to the adapter and to the attestation.
"""

from __future__ import annotations

import inspect
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from typing import Any, get_type_hints

import langchain_core
from langchain_core.callbacks import AsyncCallbackManagerForToolRun, CallbackManagerForToolRun
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool, ToolException

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

SURFACE: Surface = "langchain"
FRAMEWORK = "langchain"
ADAPTER_VERSION = mitrity.__version__
FRAMEWORK_VERSION: str | None = getattr(langchain_core, "__version__", None)

_ATTEST_RETRY_SECONDS = 30.0
_PROCESS_SESSION_ID = f"langchain-{uuid.uuid4()}"

logger = logging.getLogger("mitrity.langchain")


class LangChainGovernor:
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
        self._logger = logger or logging.getLogger("mitrity.langchain")
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

    def session_id_for(self, config: RunnableConfig | None) -> str:
        if self._session_id:
            return self._session_id
        if config:
            configurable = config.get("configurable") or {}
            thread_id = configurable.get("thread_id") if isinstance(configurable, Mapping) else None
            if isinstance(thread_id, str) and thread_id:
                return thread_id
        return _PROCESS_SESSION_ID

    def cwd(self) -> str:
        return self._cwd or os.getcwd()

    def _should_attest(self, session_id: str, digest: str) -> bool:
        with self._lock:
            if self._attested.get(session_id) == digest:
                return False
            now = time.monotonic()
            last = self._attest_attempted.get(session_id)
            # Throttle retries of a FAILED attestation only; a changed hash on
            # an attested session is re-sent at once.
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
        self,
        tool: BaseTool,
        args: tuple[Any, ...],
        kwargs: Mapping[str, Any],
        *,
        session_id: str,
        tool_use_id: str | None,
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
            tool_use_id=tool_use_id,
        )


class GovernedTool(BaseTool):
    """A ``BaseTool`` that admits every call through the MITRITY edge before running ``inner``."""

    inner: BaseTool
    governor: LangChainGovernor

    @property
    def args(self) -> dict[str, Any]:
        return self.inner.args

    @property
    def tool_call_schema(self) -> Any:
        return self.inner.tool_call_schema

    def get_input_schema(self, config: RunnableConfig | None = None) -> Any:
        return self.inner.get_input_schema(config)

    def _to_args_and_kwargs(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        # The inner tool's parsing rules apply (a single-string ``Tool`` turns
        # its input into one positional argument); the wrapper adds nothing.
        return self.inner._to_args_and_kwargs(tool_input, tool_call_id)

    def _run(
        self,
        *args: Any,
        run_manager: CallbackManagerForToolRun | None = None,
        config: RunnableConfig = RunnableConfig(),  # noqa: B008 — LangChain injects it by annotation
        **kwargs: Any,
    ) -> Any:
        session_id = self.governor.session_id_for(config)
        self.governor.ensure_attested(session_id)
        request = self.governor.request_for(
            self.inner, args, kwargs, session_id=session_id, tool_use_id=_run_id(run_manager)
        )
        verdict = self.governor.client.decide(request)
        args, kwargs = _apply_verdict(self.inner, verdict, args, kwargs)
        forwarded = _forward(self.inner._run, kwargs, run_manager, config)
        return self.inner._run(*args, **forwarded)

    async def _arun(
        self,
        *args: Any,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
        config: RunnableConfig = RunnableConfig(),  # noqa: B008
        **kwargs: Any,
    ) -> Any:
        session_id = self.governor.session_id_for(config)
        await self.governor.ensure_attested_async(session_id)
        request = self.governor.request_for(
            self.inner, args, kwargs, session_id=session_id, tool_use_id=_run_id(run_manager)
        )
        verdict = await self.governor.client.decide_async(request)
        args, kwargs = _apply_verdict(self.inner, verdict, args, kwargs)
        if type(self.inner)._arun is BaseTool._arun:
            sync_manager = run_manager.get_sync() if run_manager is not None else None
            forwarded = _forward(self.inner._run, kwargs, sync_manager, config)
            return await run_in_executor(None, self.inner._run, *args, **forwarded)
        forwarded = _forward(self.inner._arun, kwargs, run_manager, config)
        return await self.inner._arun(*args, **forwarded)


def _apply_verdict(
    tool: BaseTool, verdict: Verdict, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if not verdict.allowed:
        raise ToolException(verdict.reason)
    if verdict.updated_input is None:
        return args, kwargs
    if verdict.routed_to:
        logger.info("MITRITY routed %s to %s", tool.name, verdict.routed_to)
    updated = dict(verdict.updated_input)
    if args:
        # A positional call has exactly one place a rewrite can land. A rewrite
        # the adapter cannot place would leave the original value to run under
        # a decision made about different bytes, so it is a deny.
        key = _positional_key(tool)
        if key not in updated or set(updated) != {key}:
            raise ToolException(
                f"MITRITY rewrote the input of {tool.name} in a way the adapter cannot apply "
                f"(keys {sorted(updated)!r}); the call was blocked rather than run unchanged"
            )
        args = (updated.pop(key), *args[1:])
    return args, {**kwargs, **updated}


def _positional_key(tool: BaseTool) -> str:
    """The wire key for one positional argument: the single declared argument, else ``input``."""
    try:
        declared = list(tool.args)
    except Exception as exc:
        logger.debug(
            "MITRITY: could not introspect the arguments of %s (%r); using 'input'", tool.name, exc
        )
        declared = []
    return declared[0] if len(declared) == 1 else "input"


def _run_id(run_manager: Any) -> str | None:
    run_id = getattr(run_manager, "run_id", None)
    return str(run_id) if run_id is not None else None


def _forward(
    func: Any, kwargs: dict[str, Any], run_manager: Any, config: RunnableConfig
) -> dict[str, Any]:
    """Add ``run_manager``/``config`` to the inner call only when its signature takes them."""
    forwarded = dict(kwargs)
    try:
        parameters = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return forwarded
    if "run_manager" in parameters and run_manager is not None:
        forwarded["run_manager"] = run_manager
    config_param = _config_param(func, parameters)
    if config_param is not None:
        forwarded[config_param] = config
    return forwarded


def _config_param(func: Any, parameters: Mapping[str, inspect.Parameter]) -> str | None:
    try:
        hints = get_type_hints(func)
    except Exception:
        return None
    for name, hint in hints.items():
        if name in parameters and hint is RunnableConfig:
            return name
    return None


def govern(
    tool: BaseTool,
    *,
    client: Client | None = None,
    session_id: str | None = None,
    cwd: str | None = None,
    governor: LangChainGovernor | None = None,
) -> GovernedTool:
    """Wrap one LangChain tool so every call is admitted by the MITRITY edge first."""
    shared = (
        governor
        if governor is not None
        else LangChainGovernor(client=client, session_id=session_id, cwd=cwd)
    )
    shared.register(tool.name)
    return GovernedTool(
        inner=tool,
        governor=shared,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        return_direct=tool.return_direct,
        verbose=tool.verbose,
        callbacks=tool.callbacks,
        tags=tool.tags,
        metadata=tool.metadata,
        handle_tool_error=tool.handle_tool_error,
        handle_validation_error=tool.handle_validation_error,
        response_format=tool.response_format,
        extras=tool.extras,
    )


def govern_tools(
    tools: Iterable[BaseTool],
    *,
    client: Client | None = None,
    session_id: str | None = None,
    cwd: str | None = None,
) -> list[BaseTool]:
    """Wrap a sequence of tools with one shared governor; order is preserved."""
    shared = LangChainGovernor(client=client, session_id=session_id, cwd=cwd)
    return [govern(tool, governor=shared) for tool in tools]
