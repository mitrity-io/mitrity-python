"""Governed OpenAI Agents SDK tools: admit before the framework runs them.

The SDK executes function tools, its shell tools and the ``apply_patch``
editor in-process; hosted tools run in OpenAI's cloud. There is no hook to
install, so the adapter wraps the tools: ``govern(agent)`` returns a clone of
the agent whose tools ask the MITRITY edge first, ``govern_tools(tools)`` does
the same for a list.

Each tool type is refused the way the SDK itself refuses that type
(iag-specs ``sentinel/adapters.md``, "OpenAI Agents SDK"):

- a ``FunctionTool`` gets a tool input guardrail whose ``reject_content``
  carries the edge's reason to the model and keeps the tool from running, and
  a wrapped invoker that runs the edge's ``updated_input`` when it sends one;
- ``ShellTool`` and ``ApplyPatchTool`` go through the SDK's approval flow:
  ``needs_approval`` admits, ``on_approval`` rejects a denied call with the
  reason, the wrapped shell executor runs a rewritten action;
- ``LocalShellTool`` (deprecated upstream) has neither guardrails nor
  approvals, so its executor is wrapped and a deny is its output;
- hosted tools and ``ComputerTool`` pass through untouched and are attested
  as unhooked, so the coverage posture says what is not governed.

Coverage is exactly what is wrapped: a tool that never passed through
``govern`` is invisible to the adapter and to the attestation.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, cast

import agents
from agents import (
    Agent,
    ApplyPatchTool,
    CodeInterpreterTool,
    ComputerTool,
    FileSearchTool,
    FunctionTool,
    HostedMCPTool,
    ImageGenerationTool,
    LocalShellTool,
    RunContextWrapper,
    ShellTool,
    Tool,
    WebSearchTool,
)
from agents.editor import ApplyPatchOperation
from agents.exceptions import AgentsException, ModelBehaviorError
from agents.items import ToolApprovalItem
from agents.mcp import MCPServer
from agents.tool import (
    ApplyPatchOnApprovalFunctionResult,
    LocalShellCommandRequest,
    ShellActionRequest,
    ShellCommandRequest,
    ShellOnApprovalFunctionResult,
    ShellResult,
)
from agents.tool_context import ToolContext
from agents.tool_guardrails import (
    ToolGuardrailFunctionOutput,
    ToolInputGuardrail,
    ToolInputGuardrailData,
)

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

SURFACE: Surface = "openai_agents"
FRAMEWORK = "openai-agents"
ADAPTER_VERSION = mitrity.__version__
FRAMEWORK_VERSION: str | None = getattr(agents, "__version__", None)

GUARDRAIL_NAME = "mitrity"
_ATTEST_RETRY_SECONDS = 30.0
_PROCESS_SESSION_ID = f"openai-agents-{uuid.uuid4()}"
_VERDICT_MEMORY = 4096
_GOVERNED_MARKER = "_mitrity_governed"

HOSTED_TOOL_TYPES: tuple[type[Any], ...] = (
    WebSearchTool,
    FileSearchTool,
    CodeInterpreterTool,
    ImageGenerationTool,
    HostedMCPTool,
    ComputerTool,
)
"""Tools the adapter cannot judge: they execute in the vendor's cloud, or
(``ComputerTool``) drive a computer the SDK does not judge. Attested as
unhooked."""

logger = logging.getLogger("mitrity.openai_agents")


class MitrityDenied(AgentsException):
    """A call reached an invoker without having been admitted, and the edge denied it.

    Only raised when the guardrail the adapter installed did not run for the
    call (it was removed after ``govern()``): once that channel is gone there
    is no gentler one, and a call never runs unadmitted.
    """


@dataclass
class GovernorStats:
    """Counters a demo or a dashboard can read. Not the audit trail — the edge keeps that."""

    admitted: int = 0
    allowed: int = 0
    denied: int = 0
    held: int = 0
    unreachable: int = 0
    routed: int = 0
    attestations: int = 0


class OpenAIAgentsGovernor:
    """Shared state for a set of governed tools: the client, the session, the attestation."""

    def __init__(
        self,
        *,
        client: Client | None = None,
        session_id: str | None = None,
        cwd: str | None = None,
        gateway: MCPServer | str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client if client is not None else Client()
        self._session_id = session_id
        self._cwd = cwd
        self._gateway = gateway
        self._logger = logger or logging.getLogger("mitrity.openai_agents")
        self._hooked: list[str] = []
        self._unhooked: list[str] = []
        self._other_mcp: list[str] = []
        self._verdicts: dict[str, Verdict] = {}
        self._attested: dict[str, str] = {}
        self._attest_attempted: dict[str, float] = {}
        self._lock = threading.Lock()
        self.stats = GovernorStats()

    @property
    def client(self) -> Client:
        return self._client

    @property
    def logger(self) -> logging.Logger:
        return self._logger

    @property
    def hooked_tools(self) -> tuple[str, ...]:
        return tuple(self._hooked)

    @property
    def unhooked_tools(self) -> tuple[str, ...]:
        return tuple(self._unhooked)

    @property
    def other_mcp_servers(self) -> tuple[str, ...]:
        return tuple(self._other_mcp)

    # -------------------------------------------------------------- coverage

    def register_hooked(self, name: str) -> None:
        with self._lock:
            if name not in self._hooked:
                self._hooked.append(name)

    def register_unhooked(self, name: str) -> None:
        with self._lock:
            if name not in self._unhooked:
                self._unhooked.append(name)

    def register_mcp_server(self, name: str) -> None:
        with self._lock:
            if name not in self._other_mcp:
                self._other_mcp.append(name)

    def is_gateway(self, server: MCPServer) -> bool:
        """Whether ``server`` is the MITRITY gateway (the instance or name given as ``gateway``)."""
        if self._gateway is None:
            return False
        if isinstance(self._gateway, str):
            return server.name == self._gateway
        return server is self._gateway

    # ---------------------------------------------------------------- verdicts

    def remember(self, call_id: str | None, verdict: Verdict) -> None:
        """Keep a decision for the invoker or executor that runs the same call."""
        if not call_id:
            return
        with self._lock:
            if len(self._verdicts) >= _VERDICT_MEMORY:
                self._verdicts.pop(next(iter(self._verdicts)))
            self._verdicts[call_id] = verdict

    def peek(self, call_id: str | None) -> Verdict | None:
        if not call_id:
            return None
        with self._lock:
            return self._verdicts.get(call_id)

    def take(self, call_id: str | None) -> Verdict | None:
        if not call_id:
            return None
        with self._lock:
            return self._verdicts.pop(call_id, None)

    # ------------------------------------------------------------- attestation

    def attestation(self) -> Attestation:
        hooked = tuple(sorted(self._hooked))
        unhooked = tuple(sorted(self._unhooked))
        other = tuple(sorted(self._other_mcp))
        hashed = {
            "adapter": ADAPTER_NAME,
            "adapter_version": ADAPTER_VERSION,
            "framework": FRAMEWORK,
            "framework_version": FRAMEWORK_VERSION,
            "hooked_tools": list(hooked),
            "unhooked_exec_tools": list(unhooked),
            "disallowed_tools": [],
            "other_mcp_servers": list(other),
        }
        return Attestation(
            framework=FRAMEWORK,
            framework_version=FRAMEWORK_VERSION,
            adapter=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            hooked_tools=hooked,
            unhooked_exec_tools=unhooked,
            other_mcp_servers=other,
            config_hash=config_hash(hashed),
        )

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
        self.stats.attestations += 1

    # --------------------------------------------------------------- requests

    def session_id_for(self, ctx: RunContextWrapper[Any] | None) -> str:
        if self._session_id:
            return self._session_id
        run_config = getattr(ctx, "run_config", None)
        group_id = getattr(run_config, "group_id", None)
        if isinstance(group_id, str) and group_id:
            return group_id
        return _PROCESS_SESSION_ID

    def cwd(self) -> str:
        return self._cwd or os.getcwd()

    async def admit(
        self,
        *,
        tool_name: str,
        tool_input: Mapping[str, Any],
        ctx: RunContextWrapper[Any] | None,
        call_id: str | None,
    ) -> Verdict:
        """Attest if needed, then the two-phase decision. Never raises (G2)."""
        session_id = self.session_id_for(ctx)
        try:
            await self.ensure_attested_async(session_id)
            request = AdmitRequest(
                surface=SURFACE,
                framework_version=FRAMEWORK_VERSION,
                session_id=session_id,
                cwd=self.cwd(),
                tool_name=tool_name,
                tool_input=tool_input,
                tool_use_id=call_id,
            )
            verdict = await self._client.decide_async(request)
        except Exception as exc:
            self.stats.unreachable += 1
            self._logger.error(
                "MITRITY: admission of %s failed inside the adapter: %r", tool_name, exc
            )
            return Verdict(
                allowed=False,
                reason=(
                    f"MITRITY could not authorize {tool_name} and blocked it: {exc!r}. "
                    "This is not a policy decision — the adapter failed before the edge answered."
                ),
                error=AdmissionError(str(exc)),
            )
        self.stats.admitted += 1
        if verdict.allowed:
            self.stats.allowed += 1
            if verdict.updated_input is not None:
                self.stats.routed += 1
                if verdict.routed_to:
                    self._logger.info("MITRITY routed %s to %s", tool_name, verdict.routed_to)
        elif verdict.unreachable:
            self.stats.unreachable += 1
        elif verdict.held:
            self.stats.held += 1
        else:
            self.stats.denied += 1
        return verdict


# ---------------------------------------------------------------------------
# Function tools
# ---------------------------------------------------------------------------


def _parse_arguments(raw: str) -> dict[str, Any]:
    """The model's JSON arguments as the object the edge judges.

    A non-object (or unparseable) argument goes under ``input``.
    """
    try:
        value = json.loads(raw) if raw.strip() else {}
    except ValueError:
        return {"input": raw}
    if isinstance(value, Mapping):
        return dict(value)
    return {"input": value}


def _merge_arguments(tool_name: str, raw: str, verdict: Verdict) -> str:
    """Re-serialize the model's arguments with the edge's ``updated_input`` merged over them."""
    updated = verdict.updated_input or {}
    try:
        value = json.loads(raw) if raw.strip() else {}
    except ValueError:
        value = None
    if not isinstance(value, Mapping):
        # A non-object argument has no keys a rewrite can land on; running the
        # original under a decision made about different bytes is not an option.
        raise MitrityDenied(
            f"MITRITY rewrote the input of {tool_name} in a way the adapter cannot apply "
            f"(keys {sorted(updated)!r}); the call was blocked rather than run unchanged"
        )
    return json.dumps({**value, **updated})


def _govern_function_tool(tool: FunctionTool, governor: OpenAIAgentsGovernor) -> FunctionTool:
    if getattr(tool.on_invoke_tool, _GOVERNED_MARKER, False):
        governor.register_hooked(tool.name)
        return tool
    governor.register_hooked(tool.name)
    original_invoke = tool.on_invoke_tool

    async def guardrail_function(data: ToolInputGuardrailData) -> ToolGuardrailFunctionOutput:
        ctx = data.context
        verdict = await governor.admit(
            tool_name=tool.name,
            tool_input=_parse_arguments(ctx.tool_arguments),
            ctx=ctx,
            call_id=ctx.tool_call_id,
        )
        if verdict.allowed:
            governor.remember(ctx.tool_call_id, verdict)
            return ToolGuardrailFunctionOutput.allow()
        return ToolGuardrailFunctionOutput.reject_content(verdict.reason)

    guardrail: ToolInputGuardrail[Any] = ToolInputGuardrail(
        guardrail_function=guardrail_function, name=GUARDRAIL_NAME
    )

    async def on_invoke_tool(ctx: ToolContext[Any], input_json: str) -> Any:
        verdict = governor.take(ctx.tool_call_id)
        if verdict is None:
            # The guardrail did not run for this call: admit here. A deny has
            # no gentler channel than ending the run — and the call must not run.
            verdict = await governor.admit(
                tool_name=tool.name,
                tool_input=_parse_arguments(input_json),
                ctx=ctx,
                call_id=ctx.tool_call_id,
            )
            if not verdict.allowed:
                raise MitrityDenied(verdict.reason)
        if verdict.updated_input is not None:
            input_json = _merge_arguments(tool.name, input_json, verdict)
        return await original_invoke(ctx, input_json)

    setattr(on_invoke_tool, _GOVERNED_MARKER, True)
    existing = list(tool.tool_input_guardrails or [])
    return dataclasses.replace(
        tool, on_invoke_tool=on_invoke_tool, tool_input_guardrails=[guardrail, *existing]
    )


# ---------------------------------------------------------------------------
# Shell, apply_patch and local shell
# ---------------------------------------------------------------------------


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _developer_needs_approval(setting: Any, *args: Any) -> bool:
    """The developer's own ``needs_approval`` (a bool, or a callable that may be async)."""
    if isinstance(setting, bool):
        return setting
    if callable(setting):
        return bool(await _maybe_await(setting(*args)))
    return False


def _approval_call_id(item: ToolApprovalItem) -> str | None:
    raw = item.raw_item
    if isinstance(raw, Mapping):
        value = raw.get("call_id") or raw.get("id")
    else:
        value = getattr(raw, "call_id", None) or getattr(raw, "id", None)
    return value if isinstance(value, str) and value else None


def _fields(obj: Any) -> dict[str, Any]:
    """The fields the model set on an SDK action or operation, verbatim, ``None`` left out."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        raw = {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    elif hasattr(obj, "model_dump"):
        raw = dict(obj.model_dump())
    else:
        raw = dict(vars(obj))
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key == "ctx_wrapper" or value is None:
            continue
        out[key] = list(value) if isinstance(value, tuple) else value
    return out


def _govern_shell_tool(tool: ShellTool, governor: OpenAIAgentsGovernor) -> ShellTool:
    if getattr(tool.needs_approval, _GOVERNED_MARKER, False):
        governor.register_hooked(tool.name)
        return tool
    governor.register_hooked(tool.name)
    inner_needs = tool.needs_approval
    inner_on_approval = tool.on_approval
    inner_executor = tool.executor

    async def needs_approval(
        ctx: RunContextWrapper[Any], action: ShellActionRequest, call_id: str
    ) -> bool:
        verdict = await governor.admit(
            tool_name=tool.name, tool_input=_fields(action), ctx=ctx, call_id=call_id
        )
        governor.remember(call_id, verdict)
        if not verdict.allowed:
            # MITRITY objects: the SDK's approval step is where a rejection
            # with a reason reaches the model, and on_approval below rejects.
            return True
        return await _developer_needs_approval(inner_needs, ctx, action, call_id)

    async def on_approval(
        ctx: RunContextWrapper[Any], item: ToolApprovalItem
    ) -> ShellOnApprovalFunctionResult:
        verdict = governor.peek(_approval_call_id(item))
        if verdict is not None and not verdict.allowed:
            return {"approve": False, "reason": verdict.reason}
        if inner_on_approval is not None:
            return cast(
                ShellOnApprovalFunctionResult, await _maybe_await(inner_on_approval(ctx, item))
            )
        # MITRITY allowed and the developer asked for approval without a
        # handler: no decision here, their pending-approval interrupt stands.
        return cast(ShellOnApprovalFunctionResult, {})

    async def executor(request: ShellCommandRequest) -> str | ShellResult:
        call_id = request.data.call_id
        verdict = governor.take(call_id)
        if verdict is None:
            verdict = await governor.admit(
                tool_name=tool.name,
                tool_input=_fields(request.data.action),
                ctx=request.ctx_wrapper,
                call_id=call_id,
            )
        if not verdict.allowed:
            raise MitrityDenied(verdict.reason)
        if verdict.updated_input is not None:
            request = _rewrite_shell_request(tool.name, request, verdict)
        if inner_executor is None:
            raise ModelBehaviorError("Shell tool has no local executor configured.")
        return cast(str | ShellResult, await _maybe_await(inner_executor(request)))

    setattr(needs_approval, _GOVERNED_MARKER, True)
    return dataclasses.replace(
        tool, needs_approval=needs_approval, on_approval=on_approval, executor=executor
    )


def _rewrite_shell_request(
    tool_name: str, request: ShellCommandRequest, verdict: Verdict
) -> ShellCommandRequest:
    updated = dict(verdict.updated_input or {})
    known = {f.name for f in dataclasses.fields(request.data.action)}
    unknown = sorted(set(updated) - known)
    if unknown:
        raise MitrityDenied(
            f"MITRITY rewrote the input of {tool_name} with keys the shell action has no place "
            f"for ({unknown!r}); the call was blocked rather than run unchanged"
        )
    action = dataclasses.replace(request.data.action, **updated)
    data = dataclasses.replace(request.data, action=action)
    return dataclasses.replace(request, data=data)


def _govern_apply_patch_tool(
    tool: ApplyPatchTool, governor: OpenAIAgentsGovernor
) -> ApplyPatchTool:
    if getattr(tool.needs_approval, _GOVERNED_MARKER, False):
        governor.register_hooked(tool.name)
        return tool
    governor.register_hooked(tool.name)
    inner_needs = tool.needs_approval
    inner_on_approval = tool.on_approval

    async def needs_approval(
        ctx: RunContextWrapper[Any], operation: ApplyPatchOperation, call_id: str
    ) -> bool:
        verdict = await governor.admit(
            tool_name=tool.name, tool_input=_fields(operation), ctx=ctx, call_id=call_id
        )
        if verdict.allowed and verdict.updated_input is not None:
            # The editor receives the operation without a call id to correlate
            # a rewrite to; the original bytes must not run under a decision
            # made about different ones.
            verdict = Verdict(
                allowed=False,
                reason=(
                    f"MITRITY rewrote the input of {tool.name} "
                    f"(keys {sorted(verdict.updated_input)!r}), which the adapter cannot apply "
                    "to an apply_patch operation; the call was blocked rather than run unchanged"
                ),
                decision=verdict.decision,
                updated_input=verdict.updated_input,
                routed_to=verdict.routed_to,
            )
        governor.remember(call_id, verdict)
        if not verdict.allowed:
            return True
        return await _developer_needs_approval(inner_needs, ctx, operation, call_id)

    async def on_approval(
        ctx: RunContextWrapper[Any], item: ToolApprovalItem
    ) -> ApplyPatchOnApprovalFunctionResult:
        verdict = governor.peek(_approval_call_id(item))
        if verdict is not None and not verdict.allowed:
            return {"approve": False, "reason": verdict.reason}
        if inner_on_approval is not None:
            return cast(
                ApplyPatchOnApprovalFunctionResult,
                await _maybe_await(inner_on_approval(ctx, item)),
            )
        return cast(ApplyPatchOnApprovalFunctionResult, {})

    setattr(needs_approval, _GOVERNED_MARKER, True)
    return dataclasses.replace(tool, needs_approval=needs_approval, on_approval=on_approval)


def _govern_local_shell_tool(
    tool: LocalShellTool, governor: OpenAIAgentsGovernor
) -> LocalShellTool:
    if getattr(tool.executor, _GOVERNED_MARKER, False):
        governor.register_hooked(tool.name)
        return tool
    governor.register_hooked(tool.name)
    inner_executor = tool.executor

    async def executor(request: LocalShellCommandRequest) -> str:
        data = request.data
        action = data.action
        call_id = getattr(data, "call_id", None)
        verdict = await governor.admit(
            tool_name=tool.name,
            tool_input=_fields(action),
            ctx=request.ctx_wrapper,
            call_id=call_id if isinstance(call_id, str) else None,
        )
        if not verdict.allowed:
            # The SDK offers neither a guardrail nor an approval for this
            # deprecated tool; the executor's output is the only channel.
            return f"MITRITY denied this command and nothing was executed: {verdict.reason}"
        if verdict.updated_input is not None:
            updated = dict(verdict.updated_input)
            known = set(getattr(type(action), "model_fields", {}))
            unknown = sorted(set(updated) - known) if known else []
            if unknown:
                return (
                    f"MITRITY rewrote the input of {tool.name} with keys the action has no place "
                    f"for ({unknown!r}); the call was blocked rather than run unchanged"
                )
            new_action = action.model_copy(update=updated)
            request = LocalShellCommandRequest(
                ctx_wrapper=request.ctx_wrapper, data=data.model_copy(update={"action": new_action})
            )
        return cast(str, await _maybe_await(inner_executor(request)))

    setattr(executor, _GOVERNED_MARKER, True)
    return dataclasses.replace(tool, executor=executor)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    return name if isinstance(name, str) and name else type(tool).__name__


def govern_tools(
    tools: Iterable[Tool],
    *,
    client: Client | None = None,
    session_id: str | None = None,
    cwd: str | None = None,
    gateway: MCPServer | str | None = None,
    governor: OpenAIAgentsGovernor | None = None,
) -> list[Tool]:
    """Wrap every tool the adapter can judge; pass the rest through and attest them as unhooked."""
    shared = (
        governor
        if governor is not None
        else OpenAIAgentsGovernor(client=client, session_id=session_id, cwd=cwd, gateway=gateway)
    )
    out: list[Tool] = []
    for tool in tools:
        if isinstance(tool, FunctionTool):
            out.append(_govern_function_tool(tool, shared))
        elif isinstance(tool, ShellTool):
            out.append(_govern_shell_tool(tool, shared))
        elif isinstance(tool, ApplyPatchTool):
            out.append(_govern_apply_patch_tool(tool, shared))
        elif isinstance(tool, LocalShellTool):
            out.append(_govern_local_shell_tool(tool, shared))
        else:
            if isinstance(tool, HostedMCPTool):
                label = tool.tool_config.get("server_label") or "unknown"
                shared.register_mcp_server(f"hosted:{label}")
            shared.register_unhooked(_tool_name(tool))
            out.append(tool)
    return out


def govern(
    agent: Agent[Any],
    *,
    client: Client | None = None,
    session_id: str | None = None,
    cwd: str | None = None,
    gateway: MCPServer | str | None = None,
    governor: OpenAIAgentsGovernor | None = None,
) -> Agent[Any]:
    """A clone of ``agent`` whose tools are admitted by the MITRITY edge before they run.

    Handoff targets given as ``Agent`` objects are governed with the same
    governor, each agent once however many paths reach it, so a handoff
    cycle (a specialist handing back to the triage agent) or a diamond
    terminates on one clone per agent. A ``Handoff`` object is left as is.
    An agent ``govern`` already cloned is returned as it is. MCP servers on
    the agent are not touched: the MITRITY gateway governs its own tools,
    every other server is attested as an ungoverned path.
    """
    shared = (
        governor
        if governor is not None
        else OpenAIAgentsGovernor(client=client, session_id=session_id, cwd=cwd, gateway=gateway)
    )
    return _govern_agent(agent, shared, {})


def _govern_agent(
    agent: Agent[Any], governor: OpenAIAgentsGovernor, visited: dict[int, Agent[Any]]
) -> Agent[Any]:
    seen = visited.get(id(agent))
    if seen is not None:
        return seen
    if getattr(agent, _GOVERNED_MARKER, False):
        # A clone an earlier govern() made: its tools are wrapped and its
        # handoffs were governed with it, so it is returned as it is.
        visited[id(agent)] = agent
        return agent
    tools = govern_tools(agent.tools, governor=governor)
    for server in agent.mcp_servers:
        if not governor.is_gateway(server):
            governor.register_mcp_server(server.name)
    clone = agent.clone(tools=tools, handoffs=[])
    setattr(clone, _GOVERNED_MARKER, True)
    # Registered before the handoffs are walked, so a target that hands back
    # to this agent finds the clone instead of recursing into it again.
    visited[id(agent)] = clone
    handoffs: list[Any] = []
    for target in agent.handoffs:
        if isinstance(target, Agent):
            handoffs.append(_govern_agent(target, governor, visited))
        else:
            governor.logger.warning(
                "MITRITY: handoff %s on agent %s was built with handoff(); its agent is "
                "governed only if you governed it before building the handoff",
                getattr(target, "tool_name", None) or getattr(target, "agent_name", "?"),
                agent.name,
            )
            handoffs.append(target)
    clone.handoffs = handoffs
    return clone


__all__ = [
    "FRAMEWORK",
    "FRAMEWORK_VERSION",
    "GUARDRAIL_NAME",
    "HOSTED_TOOL_TYPES",
    "SURFACE",
    "GovernorStats",
    "MitrityDenied",
    "OpenAIAgentsGovernor",
    "govern",
    "govern_tools",
]
