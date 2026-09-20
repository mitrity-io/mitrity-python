"""Governed options for the Claude Agent SDK.

The SDK's built-in tools (``Bash``, ``Write``, ``Edit``, ``WebFetch``, ...) never
produce an MCP call, so the MITRITY gateway never sees them. This module is
what does: a ``PreToolUse`` hook that admits each call through the co-located
edge's admission API before the SDK runs it, a ``PostToolUse`` hook that keeps
the coverage claim honest, and an attestation of the runtime's posture sent at
session start.

Every guarantee in the adapter contract
(https://mitrity.com/docs/integrations/adapters) is implemented here, and the
guarantee numbers in the comments (G1–G11) refer to it.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import claude_agent_sdk
from claude_agent_sdk import ClaudeAgentOptions, HookContext, HookInput, HookJSONOutput, HookMatcher
from claude_agent_sdk.types import HookEvent, McpServerConfig, SyncHookJSONOutput

import mitrity
from mitrity.admission import (
    ADAPTER_NAME,
    ATTEST_TIMEOUT,
    EXEC_CAPABLE_TOOLS,
    HOLD_MARGIN,
    AdmissionError,
    AdmitRequest,
    Attestation,
    Client,
    SandboxPosture,
    Surface,
    config_hash,
)

SURFACE: Surface = "claude_agent_sdk"
FRAMEWORK = "claude-agent-sdk"
ADAPTER_VERSION = mitrity.__version__
FRAMEWORK_VERSION: str | None = getattr(claude_agent_sdk, "__version__", None)

DEFAULT_GATEWAY_NAME = "mitrity"
ALL_SETTING_SOURCES: tuple[str, ...] = ("user", "project", "local")

_ADMITTED_MEMORY = 4096
_ATTEST_RETRY_SECONDS = 30.0
_FRAMEWORK_HOOK_BUDGET = 600.0
_HOOK_SLACK = 30.0
_POST_HOOK_TIMEOUT = 30.0
_REASON_LIMIT = 200

logger = logging.getLogger("mitrity.claude_agent_sdk")


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
    unadmitted_executions: int = 0


@dataclass
class _Coverage:
    """What the installed options actually govern. Computed once, in ``options()``."""

    hooked: list[str]
    unhooked: list[str]
    disallowed: list[str]
    mcp_servers: list[str]
    other_mcp_servers: list[str]
    permission_mode: str
    sandbox: SandboxPosture | None
    setting_sources: list[str] | None
    strict_mcp_config: bool
    extra_unhooked: list[str] = field(default_factory=list)


class Governor:
    """Builds governed ``ClaudeAgentOptions`` and owns the hooks they install.

    One ``Governor`` per options object: ``options()`` computes the coverage the
    attestation reports, so calling it twice with different overrides would
    make the second call's attestation describe the first call's options.
    """

    def __init__(
        self,
        *,
        client: Client | None = None,
        gateway: McpServerConfig | None = None,
        gateway_name: str = DEFAULT_GATEWAY_NAME,
        hooked_tools: Sequence[str] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self._client = client if client is not None else Client()
        self._gateway = gateway
        self._gateway_name = gateway_name
        self._requested_hooks = list(hooked_tools) if hooked_tools is not None else None
        self._logger = logger or logging.getLogger("mitrity.claude_agent_sdk")
        self._coverage: _Coverage | None = None
        self._admitted: OrderedDict[str, None] = OrderedDict()
        self._attested: dict[str, str] = {}
        self._attest_attempted: dict[str, float] = {}
        self._attest_inflight: set[str] = set()
        self.stats = GovernorStats()

    @property
    def client(self) -> Client:
        return self._client

    @property
    def gateway_name(self) -> str:
        return self._gateway_name

    # ---------------------------------------------------------------- options

    def options(self, **overrides: Any) -> ClaudeAgentOptions:
        """Return ``ClaudeAgentOptions`` with MITRITY's hooks and MCP pinning applied.

        Every override is a ``ClaudeAgentOptions`` field and is passed through
        unchanged, except the ones governance has an opinion about:
        ``mcp_servers`` (merged with the gateway entry), ``hooks`` (ours run
        first; a deny from any hook wins), ``strict_mcp_config`` (default
        ``True``) and ``setting_sources`` (default ``[]``).
        """
        if self._coverage is not None:
            raise RuntimeError(
                "Governor.options() builds one options object; create a new Governor"
            )

        raw_servers = overrides.pop("mcp_servers", None)
        if raw_servers is not None and not isinstance(raw_servers, Mapping):
            raise TypeError(
                "governed_options needs mcp_servers as a dict so the MCP servers can be attested; "
                "a config file path cannot be enumerated"
            )
        mcp_servers: dict[str, Any] = dict(raw_servers or {})
        if self._gateway is not None:
            # The gateway entry owns its name: an override under the same name
            # would silently replace the governed entrance with something else.
            mcp_servers = {
                self._gateway_name: self._gateway,
                **{name: cfg for name, cfg in mcp_servers.items() if name != self._gateway_name},
            }
        governed_name = self._gateway_name if self._gateway is not None else None
        other = sorted(name for name in mcp_servers if name != governed_name)

        strict = bool(overrides.setdefault("strict_mcp_config", True))
        raw_sources = overrides.setdefault("setting_sources", [])
        sources: list[str] | None = None if raw_sources is None else [str(s) for s in raw_sources]
        if not strict:
            # Servers those files add are ungoverned paths this adapter did not
            # enumerate; naming the gap is the honest attestation (the adapter
            # contract, guarantee G6).
            for source in sources if sources is not None else ALL_SETTING_SOURCES:
                other.append(f"settings:{source}")

        tools = overrides.get("tools")
        available = (
            set(EXEC_CAPABLE_TOOLS)
            if tools is None or isinstance(tools, Mapping)
            else set(EXEC_CAPABLE_TOOLS) & {str(t) for t in tools}
        )
        disallowed = sorted(str(t) for t in (overrides.get("disallowed_tools") or []))
        requested = (
            self._requested_hooks if self._requested_hooks is not None else EXEC_CAPABLE_TOOLS
        )
        hooked = [t for t in requested if t in available]
        # Like the hook, nothing is subtracted here: the control plane subtracts
        # disallowed_tools before it raises the finding.
        unhooked = [t for t in EXEC_CAPABLE_TOOLS if t in available and t not in hooked]

        permission_mode = str(overrides.get("permission_mode") or "default")
        sandbox_raw = overrides.get("sandbox")
        sandbox = _sandbox_posture(sandbox_raw) if sandbox_raw is not None else None

        self._coverage = _Coverage(
            hooked=hooked,
            unhooked=unhooked,
            disallowed=disallowed,
            mcp_servers=sorted(mcp_servers),
            other_mcp_servers=sorted(set(other)),
            permission_mode=permission_mode,
            sandbox=sandbox,
            setting_sources=sources,
            strict_mcp_config=strict,
        )

        hooks: dict[HookEvent, list[HookMatcher]] = {}
        if hooked:
            hooks["PreToolUse"] = [
                HookMatcher(
                    matcher="|".join(hooked),
                    hooks=[self.pre_tool_use],
                    timeout=self._hook_timeout(),
                )
            ]
        hooks["PostToolUse"] = [
            HookMatcher(matcher=None, hooks=[self.post_tool_use], timeout=_POST_HOOK_TIMEOUT)
        ]
        # The Python SDK's HookEvent has no SessionStart; the first prompt of a
        # session is the earliest callback it offers.
        hooks["UserPromptSubmit"] = [
            HookMatcher(matcher=None, hooks=[self.user_prompt_submit], timeout=_POST_HOOK_TIMEOUT)
        ]
        existing = overrides.pop("hooks", None) or {}
        for event, matchers in existing.items():
            hooks.setdefault(event, []).extend(matchers)

        return ClaudeAgentOptions(mcp_servers=mcp_servers, hooks=hooks, **overrides)

    def _budgets(self) -> tuple[float, float]:
        """The hold budget one decision may spend, and the matcher timeout that contains it.

        The adapter must be the one that answers (the adapter contract, guarantee
        G3), so every term it can spend inside one PreToolUse — the attestation,
        the decision deadline, the hold wait, the hold margin — plus slack has to
        fit under the framework's own 600 s hook budget. The hold budget is what
        gives when it does not: waiting less on a human is a deny the operator
        can see; a hook the framework kills is a decision nobody made.
        """
        cfg = self._client.config
        fixed = ATTEST_TIMEOUT + cfg.timeout + HOLD_MARGIN + _HOOK_SLACK
        hold = min(cfg.hold_timeout, max(0.0, _FRAMEWORK_HOOK_BUDGET - fixed))
        return hold, min(_FRAMEWORK_HOOK_BUDGET, fixed + hold)

    def _hook_timeout(self) -> float:
        return self._budgets()[1]

    @property
    def hold_budget(self) -> float:
        """Seconds one decision may wait on a human approval, after the budget fit."""
        return self._budgets()[0]

    # ------------------------------------------------------------ attestation

    def attestation(self, permission_mode: str | None = None) -> Attestation:
        """The ``RuntimeAttestation`` for the options this governor built."""
        cov = self._require_coverage()
        mode = permission_mode or cov.permission_mode
        unhooked = sorted(set(cov.unhooked) | set(cov.extra_unhooked))
        hashed = {
            "adapter": ADAPTER_NAME,
            "adapter_version": ADAPTER_VERSION,
            "framework": FRAMEWORK,
            "framework_version": FRAMEWORK_VERSION,
            "hooked_tools": sorted(cov.hooked),
            "unhooked_exec_tools": unhooked,
            "disallowed_tools": cov.disallowed,
            "mcp_servers": cov.mcp_servers,
            "other_mcp_servers": cov.other_mcp_servers,
            "permission_mode": mode,
            "sandbox": cov.sandbox.to_wire() if cov.sandbox is not None else None,
            "setting_sources": cov.setting_sources,
            "strict_mcp_config": cov.strict_mcp_config,
        }
        return Attestation(
            framework=FRAMEWORK,
            framework_version=FRAMEWORK_VERSION,
            adapter=ADAPTER_NAME,
            adapter_version=ADAPTER_VERSION,
            hooked_tools=tuple(sorted(cov.hooked)),
            unhooked_exec_tools=tuple(unhooked),
            disallowed_tools=tuple(cov.disallowed),
            other_mcp_servers=tuple(cov.other_mcp_servers),
            permission_mode=mode,
            sandbox=cov.sandbox,
            config_hash=config_hash(hashed),
        )

    async def _ensure_attested(self, data: Mapping[str, Any]) -> None:
        """Attest once per session and config hash; re-attest when either changes.

        Best effort in the hook's sense: a failed attestation is logged, never
        a reason to block a call, because its absence is the signal the control
        plane is built to notice. It is retried on later events, but not more
        often than every 30 s per session.
        """
        session_id = str(data.get("session_id") or "")
        mode = data.get("permission_mode")
        attestation = self.attestation(mode if isinstance(mode, str) and mode else None)
        digest = attestation.config_hash or ""
        if self._attested.get(session_id) == digest:
            return
        now = time.monotonic()
        last = self._attest_attempted.get(session_id)
        # Throttle retries of a FAILED attestation only; a changed hash on an
        # attested session is re-sent at once.
        if (
            last is not None
            and now - last < _ATTEST_RETRY_SECONDS
            and session_id not in self._attested
        ):
            return
        if session_id in self._attest_inflight:
            # A burst of concurrent hook calls at session start attests once, not once per call.
            return
        self._attest_attempted[session_id] = now
        self._attest_inflight.add(session_id)
        try:
            await self._client.attest_async(attestation)
        except AdmissionError as exc:
            self._logger.warning("MITRITY: could not report the runtime posture: %s", exc)
            return
        finally:
            self._attest_inflight.discard(session_id)
        self._attested[session_id] = digest
        self.stats.attestations += 1

    # ------------------------------------------------------------------ hooks

    async def pre_tool_use(
        self, input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Admit one built-in tool call before the SDK runs it."""
        data = cast("Mapping[str, Any]", input_data)
        tool_name = str(data.get("tool_name") or "")
        call_id = tool_use_id or _optional_str(data.get("tool_use_id"))
        try:
            await self._ensure_attested(data)
            if not tool_name:
                return _deny("MITRITY blocked this action: the hook payload carried no tool_name")
            if tool_name.startswith("mcp__"):
                # An MCP tool is the gateway's call to judge (or another server's
                # ungoverned one, already attested); never admitted twice.
                return _allow()
            raw_input = data.get("tool_input")
            tool_input: dict[str, Any] = dict(raw_input) if isinstance(raw_input, Mapping) else {}
            request = AdmitRequest(
                surface=SURFACE,
                framework_version=FRAMEWORK_VERSION,
                session_id=_optional_str(data.get("session_id")),
                cwd=_optional_str(data.get("cwd")),
                tool_name=tool_name,
                tool_input=tool_input,
                tool_use_id=call_id,
            )
            verdict = await self._client.decide_async(request, hold_timeout=self.hold_budget)
        except Exception as exc:
            self.stats.unreachable += 1
            self._logger.error(
                "MITRITY: admission of %s failed inside the adapter: %r", tool_name, exc
            )
            return _deny(
                f"MITRITY could not authorize this action and blocked it: {_bound(repr(exc))}. "
                "This is not a policy decision — the adapter failed before the edge answered."
            )

        self._remember(call_id)
        self.stats.admitted += 1
        if verdict.allowed:
            self.stats.allowed += 1
            if verdict.updated_input is not None:
                # Routed (or rewritten): the merged input is what runs, and the
                # explicit allow keeps a human from being prompted about a relay
                # command that carries a ticket (the adapter contract, guarantee G10).
                self.stats.routed += 1
                merged = {**tool_input, **verdict.updated_input}
                if verdict.routed_to:
                    self._logger.info("MITRITY routed %s to %s", tool_name, verdict.routed_to)
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "updatedInput": merged,
                    }
                }
            return _allow()

        if verdict.unreachable:
            self.stats.unreachable += 1
        elif verdict.held:
            self.stats.held += 1
        else:
            self.stats.denied += 1
        output = _deny(verdict.reason)
        if verdict.held:
            output["systemMessage"] = (
                f"MITRITY held {tool_name} for human approval and it was not approved within "
                f"the hold budget ({int(self.hold_budget)}s)."
            )
        return output

    async def post_tool_use(
        self, input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """Keep the coverage claim honest: a call that ran without being admitted is a gap."""
        data = cast("Mapping[str, Any]", input_data)
        tool_name = str(data.get("tool_name") or "")
        call_id = tool_use_id or _optional_str(data.get("tool_use_id"))
        try:
            if call_id is not None and self._admitted.pop(call_id, _MISSING) is not _MISSING:
                return _allow()
            if not tool_name or tool_name.startswith("mcp__"):
                return _allow()
            cov = self._coverage
            if cov is None or tool_name not in EXEC_CAPABLE_TOOLS:
                return _allow()
            self.stats.unadmitted_executions += 1
            if tool_name in cov.hooked:
                self._logger.warning(
                    "MITRITY: %s ran without passing through the admission hook (tool_use_id=%s)",
                    tool_name,
                    call_id,
                )
            elif tool_name not in cov.unhooked and tool_name not in cov.extra_unhooked:
                cov.extra_unhooked.append(tool_name)
                self._logger.warning(
                    "MITRITY: %s executed unhooked; re-attesting coverage", tool_name
                )
            await self._ensure_attested(data)
        except Exception as exc:
            self._logger.error("MITRITY: post-tool bookkeeping failed: %r", exc)
        return _allow()

    async def user_prompt_submit(
        self, input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        """The session-start trigger the Python SDK offers: attest on the first prompt."""
        data = cast("Mapping[str, Any]", input_data)
        try:
            await self._ensure_attested(data)
        except Exception as exc:
            self._logger.error("MITRITY: attestation failed: %r", exc)
        return _allow()

    # -------------------------------------------------------------- plumbing

    def _require_coverage(self) -> _Coverage:
        if self._coverage is None:
            raise RuntimeError("call Governor.options() before asking for the attestation")
        return self._coverage

    def _remember(self, call_id: str | None) -> None:
        if call_id is None:
            return
        self._admitted[call_id] = None
        while len(self._admitted) > _ADMITTED_MEMORY:
            self._admitted.popitem(last=False)


_MISSING = object()


def _bound(text: str) -> str:
    """Bound text that reaches the model's context, as the client bounds error bodies."""
    return text if len(text) <= _REASON_LIMIT else text[:_REASON_LIMIT] + "…"


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _allow() -> SyncHookJSONOutput:
    # Silent: no permissionDecision, so the developer's own permission flow
    # continues unchanged (the adapter contract, guarantee G10).
    return {}


def _deny(reason: str) -> SyncHookJSONOutput:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _sandbox_posture(raw: Any) -> SandboxPosture:
    settings = cast("Mapping[str, Any]", raw) if isinstance(raw, Mapping) else {}
    return SandboxPosture(
        enabled=_optional_bool(settings.get("enabled")),
        allow_unsandboxed_commands=_optional_bool(settings.get("allowUnsandboxedCommands")),
        fail_if_unavailable=_optional_bool(settings.get("failIfUnavailable")),
    )


def _optional_bool(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def governed_options(
    *,
    client: Client | None = None,
    gateway: McpServerConfig | None = None,
    gateway_name: str = DEFAULT_GATEWAY_NAME,
    hooked_tools: Sequence[str] | None = None,
    **overrides: Any,
) -> ClaudeAgentOptions:
    """``ClaudeAgentOptions`` whose built-in tools are admitted by the MITRITY edge.

    ``gateway`` is the co-located gateway's MCP server config (usually a stdio
    entry); it becomes ``mcp_servers[gateway_name]`` and the only governed MCP
    path. ``hooked_tools`` narrows the admitted built-ins (default: every
    execution-capable tool); anything left out is attested as unhooked.
    ``overrides`` are ordinary ``ClaudeAgentOptions`` fields.
    """
    governor = Governor(
        client=client, gateway=gateway, gateway_name=gateway_name, hooked_tools=hooked_tools
    )
    return governor.options(**overrides)
