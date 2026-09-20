"""Governed OpenAI Agents SDK tools: admit before the framework runs them.

The SDK executes function tools, its shell tools and the ``apply_patch``
editor in-process; hosted tools run in OpenAI's cloud. There is no hook to
install, so the adapter wraps the tools: ``govern(agent)`` returns a clone of
the agent whose tools ask the MITRITY edge first, ``govern_tools(tools)`` does
the same for a list.

Each tool type is refused the way the SDK itself refuses that type (the
adapter contract, "OpenAI Agents SDK" — https://mitrity.com/docs/integrations/adapters):

- a ``FunctionTool`` gets a tool input guardrail whose ``reject_content``
  carries the edge's reason to the model and keeps the tool from running, and
  a wrapped invoker that runs the edge's ``updated_input`` when it sends one;
- ``ShellTool`` and ``ApplyPatchTool`` go through the SDK's approval flow:
  ``needs_approval`` admits, ``on_approval`` rejects a denied call with the
  reason — and an approval it cannot tie to a decision. The execution
  channel, the shell executor and the apply_patch editor, is wrapped as
  well: a call the SDK runs without asking (an approved call on a resumed
  run, a standing approval) is admitted there, and a rewritten shell action
  is what the developer's executor receives;
- ``LocalShellTool`` (deprecated upstream) has neither guardrails nor
  approvals, so its executor is wrapped and a deny is its output;
- hosted tools, ``ComputerTool`` and a ``ShellTool`` whose ``environment`` is
  a hosted container pass through untouched and are attested as unhooked,
  so the coverage posture says what is not governed.

Judged bytes are the executed bytes. A decision is remembered together with
a digest of the input it was made for, keyed by the tool and the SDK's call
id; the channel that runs the call reuses it only for that exact input, and
an execution whose input differs from what the edge judged is blocked rather
than run under that decision.

Coverage is exactly what is wrapped: a tool that never passed through
``govern`` is invisible to the adapter and to the attestation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
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
_REASON_LIMIT = 200

_MISMATCH_REASON = (
    "MITRITY judged a different input for {tool} (call {call_id}) than the one about to run; "
    "the call was blocked rather than run under a decision made about other bytes"
)
_UNCORRELATED_REASON = (
    "MITRITY could not correlate this {tool} approval (call {call_id}) to a decision it made "
    "and rejected it; the call was blocked rather than run unjudged"
)

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
unhooked. A tool type outside this tuple and the governed types is passed
through and attested as unhooked too, with a warning: the honest default for
a future SDK tool is "not governed", never silence."""

logger = logging.getLogger("mitrity.openai_agents")


class MitrityDenied(AgentsException):
    """A call reached an execution channel and MITRITY would not let it run there.

    Raised where the SDK offers no gentler channel than ending the step: an
    invoker or executor the guardrail or approval did not gate (the guardrail
    was removed after ``govern()``, the call was approved on a resumed run),
    an execution whose input differs from the judged one, an apply_patch
    operation the wrapped editor refuses. A call never runs unadmitted.
    """


@dataclass
class GovernorStats:
    """Counters a demo or a dashboard can read. Not the audit trail — the edge keeps that.

    Updated under the governor's lock, so concurrent tool calls count exactly.
    """

    admitted: int = 0
    allowed: int = 0
    denied: int = 0
    held: int = 0
    unreachable: int = 0
    routed: int = 0
    attestations: int = 0
    uncorrelated: int = 0
    """Approval items the adapter could not correlate to a decision; each was rejected."""
    mismatched: int = 0
    """Executions whose input differed from the judged input; each was blocked."""


@dataclass(frozen=True)
class Recalled:
    """What the verdict memory holds for a call and the input about to run.

    ``verdict`` is the decision made for exactly these bytes, ``None`` when
    nothing judged the call. ``other_input`` says the call *was* judged — for
    a different input — so the caller must not run under that decision.
    """

    verdict: Verdict | None = None
    other_input: bool = False


def input_digest(tool_input: Mapping[str, Any]) -> str | None:
    """SHA-256 of the canonical JSON of a tool input: sorted keys, no whitespace.

    Deliberately not RFC 8785: that canonicalization refuses floats, and a
    function tool may well take one. Sorted, compact ``json.dumps`` is
    deterministic for the same value, which is all a same-process comparison
    needs. ``None`` when the input cannot be serialized at all — nothing is
    remembered for it, and a comparison against it fails closed.
    """
    try:
        text = json.dumps(tool_input, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bound(text: str) -> str:
    """Bound text that reaches the model's context, as the client bounds error bodies."""
    return text if len(text) <= _REASON_LIMIT else text[:_REASON_LIMIT] + "…"


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
        # (tool name, call id) -> digest of the judged input -> the verdict.
        # One call id carries one input, except apply_patch, whose call may
        # carry several operations; each is judged and remembered on its own.
        self._judged: OrderedDict[tuple[str, str], dict[str, Verdict]] = OrderedDict()
        # (tool name, digest) -> the key above, for a channel with no call id.
        self._by_input: dict[tuple[str, str], tuple[str, str]] = {}
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

    def remember(
        self, tool_name: str, call_id: str | None, tool_input: Mapping[str, Any], verdict: Verdict
    ) -> None:
        """Keep a decision, with the digest of the judged input, for the channel that runs the call.

        A deny replaces every earlier verdict for the call: nothing allowed
        under this id before the refusal survives it.
        """
        digest = input_digest(tool_input)
        if not call_id or digest is None:
            return
        key = (tool_name, call_id)
        with self._lock:
            entry = self._judged.get(key)
            if entry is None or not verdict.allowed:
                self._forget_locked(key)
                entry = {}
                self._judged[key] = entry
            entry[digest] = verdict
            self._by_input[(tool_name, digest)] = key
            while len(self._judged) > _VERDICT_MEMORY:
                self._forget_locked(next(iter(self._judged)))

    def _forget_locked(self, key: tuple[str, str]) -> None:
        entry = self._judged.pop(key, None)
        for digest in entry or ():
            if self._by_input.get((key[0], digest)) == key:
                del self._by_input[(key[0], digest)]

    def peek(self, tool_name: str, call_id: str | None) -> Verdict | None:
        """The verdict standing for a call, a deny winning over an allow; ``None`` when unjudged."""
        if not call_id:
            return None
        with self._lock:
            entry = self._judged.get((tool_name, call_id))
            if not entry:
                return None
            for verdict in entry.values():
                if not verdict.allowed:
                    return verdict
            return next(iter(entry.values()))

    def peek_input(
        self, tool_name: str, call_id: str | None, tool_input: Mapping[str, Any]
    ) -> Verdict | None:
        """The verdict made for exactly this call and input, left in place."""
        digest = input_digest(tool_input)
        if not call_id or digest is None:
            return None
        with self._lock:
            entry = self._judged.get((tool_name, call_id))
            return entry.get(digest) if entry else None

    def take(self, tool_name: str, call_id: str | None, tool_input: Mapping[str, Any]) -> Recalled:
        """Consume the verdict made for exactly this call and input.

        A call judged for other bytes is reported as such and forgotten
        entirely: nothing under that id runs without a fresh decision.
        """
        if not call_id:
            return Recalled()
        digest = input_digest(tool_input)
        key = (tool_name, call_id)
        with self._lock:
            entry = self._judged.get(key)
            if entry is None:
                return Recalled()
            verdict = entry.get(digest) if digest is not None else None
            if verdict is None or digest is None:
                self._forget_locked(key)
                return Recalled(other_input=True)
            del entry[digest]
            if self._by_input.get((tool_name, digest)) == key:
                del self._by_input[(tool_name, digest)]
            if not entry:
                del self._judged[key]
            return Recalled(verdict=verdict)

    def take_input(self, tool_name: str, tool_input: Mapping[str, Any]) -> Verdict | None:
        """Consume the verdict made for this input on a channel that carries no call id."""
        digest = input_digest(tool_input)
        if digest is None:
            return None
        with self._lock:
            key = self._by_input.pop((tool_name, digest), None)
            if key is None:
                return None
            entry = self._judged.get(key)
            verdict = entry.pop(digest, None) if entry else None
            if entry is not None and not entry:
                del self._judged[key]
            return verdict

    def reject_uncorrelated(self, tool_name: str, call_id: str | None) -> str:
        """The reason an approval item the adapter cannot tie to a decision is rejected with.

        Counted and logged (tool name and call id only): the edge never saw
        this call, so the adapter's log is the only record of the refusal.
        """
        self._bump("uncorrelated")
        self._logger.warning(
            "MITRITY: rejected an approval for %s (call %s) it could not correlate to a decision",
            tool_name,
            call_id or "unknown",
        )
        return _UNCORRELATED_REASON.format(tool=tool_name, call_id=call_id or "unknown")

    def mismatch(self, tool_name: str, call_id: str | None) -> MitrityDenied:
        """The refusal for an execution whose input is not the one the edge judged."""
        self._bump("mismatched")
        self._logger.warning(
            "MITRITY: blocked %s (call %s): its input is not the input the edge judged",
            tool_name,
            call_id or "unknown",
        )
        return MitrityDenied(_MISMATCH_REASON.format(tool=tool_name, call_id=call_id or "unknown"))

    def _bump(self, counter: str) -> None:
        with self._lock:
            setattr(self.stats, counter, getattr(self.stats, counter) + 1)

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
        self._bump("attestations")

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
            self._bump("unreachable")
            self._logger.error(
                "MITRITY: admission of %s failed inside the adapter: %r", tool_name, exc
            )
            return Verdict(
                allowed=False,
                reason=(
                    f"MITRITY could not authorize {tool_name} and blocked it: "
                    f"{_bound(repr(exc))}. "
                    "This is not a policy decision — the adapter failed before the edge answered."
                ),
                error=AdmissionError(str(exc)),
            )
        self._bump("admitted")
        if verdict.allowed:
            self._bump("allowed")
            if verdict.updated_input is not None:
                self._bump("routed")
                if verdict.routed_to:
                    self._logger.info("MITRITY routed %s to %s", tool_name, verdict.routed_to)
        elif verdict.unreachable:
            self._bump("unreachable")
        elif verdict.held:
            self._bump("held")
        else:
            self._bump("denied")
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
        tool_input = _parse_arguments(ctx.tool_arguments)
        verdict = await governor.admit(
            tool_name=tool.name, tool_input=tool_input, ctx=ctx, call_id=ctx.tool_call_id
        )
        # Remembered whatever it says: the invoker reuses an allow for the same
        # bytes, and a deny replaces any allow an earlier call under this id left.
        governor.remember(tool.name, ctx.tool_call_id, tool_input, verdict)
        if verdict.allowed:
            return ToolGuardrailFunctionOutput.allow()
        return ToolGuardrailFunctionOutput.reject_content(verdict.reason)

    guardrail: ToolInputGuardrail[Any] = ToolInputGuardrail(
        guardrail_function=guardrail_function, name=GUARDRAIL_NAME
    )

    async def on_invoke_tool(ctx: ToolContext[Any], input_json: str) -> Any:
        tool_input = _parse_arguments(input_json)
        recalled = governor.take(tool.name, ctx.tool_call_id, tool_input)
        if recalled.other_input:
            raise governor.mismatch(tool.name, ctx.tool_call_id)
        verdict = recalled.verdict
        if verdict is None:
            # The guardrail did not run for this call: admit here. A deny has
            # no gentler channel than ending the run — and the call must not run.
            verdict = await governor.admit(
                tool_name=tool.name, tool_input=tool_input, ctx=ctx, call_id=ctx.tool_call_id
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
    """The ``call_id`` of the call an approval item stands for.

    Read the way the SDK reads it for shell and apply_patch calls — the
    ``call_id`` of the raw item, mapping or object — and nothing else: the
    item ``id`` is not a call id, and guessing one would correlate the
    approval to a decision made about another call.
    """
    raw = item.raw_item
    value = raw.get("call_id") if isinstance(raw, Mapping) else getattr(raw, "call_id", None)
    return value if isinstance(value, str) and value else None


async def _approval_decision(
    tool_name: str,
    governor: OpenAIAgentsGovernor,
    inner_on_approval: Any,
    ctx: RunContextWrapper[Any],
    item: ToolApprovalItem,
) -> Any:
    """The adapter's ``on_approval``: reject what MITRITY denied or cannot place, else defer.

    An approval item the adapter cannot correlate to a decision — no call id
    on the item, or a call nothing judged — is rejected, never left to the
    developer's handler or to the pending interrupt: an auto-approving host
    would otherwise run a call the edge never saw.
    """
    call_id = _approval_call_id(item)
    verdict = governor.peek(tool_name, call_id)
    if verdict is None:
        return {"approve": False, "reason": governor.reject_uncorrelated(tool_name, call_id)}
    if not verdict.allowed:
        return {"approve": False, "reason": verdict.reason}
    if inner_on_approval is not None:
        return await _maybe_await(inner_on_approval(ctx, item))
    # MITRITY allowed and the developer asked for approval without a
    # handler: no decision here, their pending-approval interrupt stands.
    return {}


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


def _shell_environment_type(tool: ShellTool) -> str:
    """Where the SDK runs this shell tool's commands: ``local``, or a hosted container type."""
    environment = tool.environment
    kind = environment.get("type") if environment else None
    return kind if isinstance(kind, str) and kind else "local"


def _govern_shell_tool(tool: ShellTool, governor: OpenAIAgentsGovernor) -> ShellTool:
    if getattr(tool.needs_approval, _GOVERNED_MARKER, False):
        governor.register_hooked(tool.name)
        return tool
    environment = _shell_environment_type(tool)
    if environment != "local":
        # The commands run in OpenAI's container: no executor, no approval
        # flow, nothing in-process the adapter can judge. Like the hosted
        # tools it passes through, and the attestation names the gap.
        governor.logger.warning(
            "MITRITY: %s runs in a hosted %s environment the adapter cannot judge; it passes "
            "through ungoverned and is attested as unhooked",
            tool.name,
            environment,
        )
        governor.register_unhooked(tool.name)
        return tool
    governor.register_hooked(tool.name)
    inner_needs = tool.needs_approval
    inner_on_approval = tool.on_approval
    inner_executor = tool.executor

    async def needs_approval(
        ctx: RunContextWrapper[Any], action: ShellActionRequest, call_id: str
    ) -> bool:
        tool_input = _fields(action)
        # The SDK evaluates needs_approval more than once for one call on a
        # resumed turn; the same call and the same bytes get the decision
        # already made for them, not a second admission.
        verdict = governor.peek_input(tool.name, call_id, tool_input)
        if verdict is None:
            verdict = await governor.admit(
                tool_name=tool.name, tool_input=tool_input, ctx=ctx, call_id=call_id
            )
            governor.remember(tool.name, call_id, tool_input, verdict)
        if not verdict.allowed:
            # MITRITY objects: the SDK's approval step is where a rejection
            # with a reason reaches the model, and on_approval below rejects.
            return True
        return await _developer_needs_approval(inner_needs, ctx, action, call_id)

    async def on_approval(
        ctx: RunContextWrapper[Any], item: ToolApprovalItem
    ) -> ShellOnApprovalFunctionResult:
        return cast(
            ShellOnApprovalFunctionResult,
            await _approval_decision(tool.name, governor, inner_on_approval, ctx, item),
        )

    async def executor(request: ShellCommandRequest) -> str | ShellResult:
        call_id = request.data.call_id
        tool_input = _fields(request.data.action)
        recalled = governor.take(tool.name, call_id, tool_input)
        if recalled.other_input:
            raise governor.mismatch(tool.name, call_id)
        verdict = recalled.verdict
        if verdict is None:
            # Nothing judged this call — an approved call on a resumed run
            # skips needs_approval — so the executor is the gate.
            verdict = await governor.admit(
                tool_name=tool.name,
                tool_input=tool_input,
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


def _refuse_apply_patch_rewrite(tool_name: str, verdict: Verdict) -> Verdict:
    """An allow that rewrites an apply_patch operation is a deny.

    No editor channel can apply a rewrite: the editor receives the operation
    alone, and the original bytes must not run under a decision made about
    different ones.
    """
    if not verdict.allowed or verdict.updated_input is None:
        return verdict
    return Verdict(
        allowed=False,
        reason=(
            f"MITRITY rewrote the input of {tool_name} "
            f"(keys {sorted(verdict.updated_input)!r}), which the adapter cannot apply "
            "to an apply_patch operation; the call was blocked rather than run unchanged"
        ),
        decision=verdict.decision,
        updated_input=verdict.updated_input,
        routed_to=verdict.routed_to,
    )


class _GovernedEditor:
    """An ``ApplyPatchEditor`` that admits every operation before the developer's editor runs it.

    The SDK calls the editor with the operation alone — no call id — so the
    decision ``needs_approval`` made is found by the operation's digest. An
    operation nothing judged (an approved call on a resumed run skips
    ``needs_approval``) is admitted here. A deny, a rewrite and an editor
    that cannot perform the operation each raise ``MitrityDenied``: the SDK
    reports the failure to the model, and nothing is applied.
    """

    def __init__(self, inner: Any, tool_name: str, governor: OpenAIAgentsGovernor) -> None:
        self._inner = inner
        self._tool_name = tool_name
        self._governor = governor

    @property
    def inner(self) -> Any:
        return self._inner

    def can_perform(self, method: str) -> bool:
        return callable(getattr(self._inner, method, None))

    async def create_file(self, operation: ApplyPatchOperation) -> Any:
        return await self._apply("create_file", operation)

    async def update_file(self, operation: ApplyPatchOperation) -> Any:
        return await self._apply("update_file", operation)

    async def delete_file(self, operation: ApplyPatchOperation) -> Any:
        return await self._apply("delete_file", operation)

    async def _apply(self, method: str, operation: ApplyPatchOperation) -> Any:
        tool_input = _fields(operation)
        verdict = self._governor.take_input(self._tool_name, tool_input)
        if verdict is None:
            verdict = await self._governor.admit(
                tool_name=self._tool_name,
                tool_input=tool_input,
                ctx=operation.ctx_wrapper,
                call_id=None,
            )
            verdict = _refuse_apply_patch_rewrite(self._tool_name, verdict)
        if not verdict.allowed:
            raise MitrityDenied(verdict.reason)
        if not self.can_perform(method):
            raise MitrityDenied(
                f"MITRITY blocked {self._tool_name}: the tool has no editor able to "
                f"{method.replace('_', ' ')}, so nothing was applied. "
                "This is not a policy decision."
            )
        return await _maybe_await(getattr(self._inner, method)(operation))


_EDITOR_METHODS = ("create_file", "update_file", "delete_file")


def _govern_apply_patch_tool(
    tool: ApplyPatchTool, governor: OpenAIAgentsGovernor
) -> ApplyPatchTool:
    if getattr(tool.needs_approval, _GOVERNED_MARKER, False):
        governor.register_hooked(tool.name)
        return tool
    governor.register_hooked(tool.name)
    inner_needs = tool.needs_approval
    inner_on_approval = tool.on_approval
    editor = _GovernedEditor(tool.editor, tool.name, governor)
    missing = [method for method in _EDITOR_METHODS if not editor.can_perform(method)]
    if missing:
        governor.logger.warning(
            "MITRITY: %s has no editor able to %s; those operations will be blocked",
            tool.name,
            ", ".join(method.replace("_", " ") for method in missing),
        )

    async def needs_approval(
        ctx: RunContextWrapper[Any], operation: ApplyPatchOperation, call_id: str
    ) -> bool:
        tool_input = _fields(operation)
        verdict = governor.peek_input(tool.name, call_id, tool_input)
        if verdict is None:
            verdict = await governor.admit(
                tool_name=tool.name, tool_input=tool_input, ctx=ctx, call_id=call_id
            )
            verdict = _refuse_apply_patch_rewrite(tool.name, verdict)
            governor.remember(tool.name, call_id, tool_input, verdict)
        if not verdict.allowed:
            return True
        return await _developer_needs_approval(inner_needs, ctx, operation, call_id)

    async def on_approval(
        ctx: RunContextWrapper[Any], item: ToolApprovalItem
    ) -> ApplyPatchOnApprovalFunctionResult:
        return cast(
            ApplyPatchOnApprovalFunctionResult,
            await _approval_decision(tool.name, governor, inner_on_approval, ctx, item),
        )

    setattr(needs_approval, _GOVERNED_MARKER, True)
    return dataclasses.replace(
        tool, needs_approval=needs_approval, on_approval=on_approval, editor=editor
    )


def _rewrite_local_shell_request(
    tool_name: str, request: LocalShellCommandRequest, updated: Mapping[str, Any]
) -> LocalShellCommandRequest | str:
    """The request with ``updated`` applied to its action, or the reason the rewrite was refused.

    The action is one of the API's pydantic models; a rewrite is applied only
    when its fields can be enumerated and every rewritten key is one of them.
    Anything less — an action the adapter cannot introspect or copy, a key
    the action has no place for — refuses the rewrite and blocks the call:
    the original command must not run under a decision made about the
    rewritten one.
    """
    data = request.data
    action = data.action
    known = set(getattr(type(action), "model_fields", None) or ())
    if not known:
        return (
            f"MITRITY rewrote the input of {tool_name} (keys {sorted(updated)!r}) but the "
            "adapter cannot introspect the action to apply it; the call was blocked rather "
            "than run unchanged"
        )
    unknown = sorted(set(updated) - known)
    if unknown:
        return (
            f"MITRITY rewrote the input of {tool_name} with keys the action has no place "
            f"for ({unknown!r}); the call was blocked rather than run unchanged"
        )
    try:
        new_action = action.model_copy(update=dict(updated))
        new_data = data.model_copy(update={"action": new_action})
    except Exception as exc:
        return (
            f"MITRITY rewrote the input of {tool_name} but the adapter could not apply it "
            f"({_bound(repr(exc))}); the call was blocked rather than run unchanged"
        )
    return LocalShellCommandRequest(ctx_wrapper=request.ctx_wrapper, data=new_data)


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
        call_id = getattr(data, "call_id", None)
        verdict = await governor.admit(
            tool_name=tool.name,
            tool_input=_fields(data.action),
            ctx=request.ctx_wrapper,
            call_id=call_id if isinstance(call_id, str) else None,
        )
        if not verdict.allowed:
            # The SDK offers neither a guardrail nor an approval for this
            # deprecated tool; the executor's output is the only channel.
            return f"MITRITY denied this command and nothing was executed: {verdict.reason}"
        if verdict.updated_input is not None:
            rewritten = _rewrite_local_shell_request(tool.name, request, verdict.updated_input)
            if isinstance(rewritten, str):
                return rewritten
            request = rewritten
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
            name = _tool_name(tool)
            if isinstance(tool, HostedMCPTool):
                label = tool.tool_config.get("server_label") or "unknown"
                shared.register_mcp_server(f"hosted:{label}")
            if not isinstance(tool, HOSTED_TOOL_TYPES):
                shared.logger.warning(
                    "MITRITY: %s (%s) is a tool type this adapter does not know; it passes "
                    "through ungoverned and is attested as unhooked",
                    name,
                    type(tool).__name__,
                )
            shared.register_unhooked(name)
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
    "Recalled",
    "govern",
    "govern_tools",
    "input_digest",
]
