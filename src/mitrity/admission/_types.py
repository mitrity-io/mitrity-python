"""Wire types of the admission API, as the adapter sees them."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ._errors import AdmissionError, AdmissionProtocolError

Surface = Literal[
    "claude_code", "claude_agent_sdk", "langchain", "openai_agents", "crewai", "custom"
]
"""The calling framework, as the request names it. Not the audit surface (always ``agent_hook``)."""

DecisionKind = Literal["allow", "deny", "held"]

FRAMEWORK_FOR_SURFACE: Mapping[str, str] = {
    "claude_code": "claude-code",
    "claude_agent_sdk": "claude-agent-sdk",
    "langchain": "langchain",
    "openai_agents": "openai-agents",
    "crewai": "crewai",
    "custom": "custom",
}
"""Request ``surface`` (underscores) to ``RuntimeAttestation.framework`` (hyphens)."""

EXEC_CAPABLE_TOOLS: tuple[str, ...] = (
    "Bash",
    "Write",
    "Edit",
    "MultiEdit",
    "NotebookEdit",
    "WebFetch",
    "WebSearch",
)
"""The Claude Code / Agent SDK built-in tools that can change something outside the model's
context. The inventory ``unhooked_exec_tools`` is measured against; ``Read``, ``Glob`` and
``Grep`` are deliberately absent, as in the hook."""

ADAPTER_NAME = "mitrity-python"


@dataclass(frozen=True)
class AdmitRequest:
    """One ``POST /v1/admit`` body.

    ``tool_name`` and ``tool_input`` are the framework's, verbatim: nothing is
    renamed or dropped, so a policy that matches a key on the MCP entrance
    matches the same key here.
    """

    tool_name: str
    tool_input: Mapping[str, Any]
    surface: Surface = "custom"
    framework_version: str | None = None
    session_id: str | None = None
    cwd: str | None = None
    tool_use_id: str | None = None
    hold_timeout_seconds: int | None = None

    def to_wire(self) -> dict[str, Any]:
        if self.surface not in FRAMEWORK_FOR_SURFACE:
            raise AdmissionProtocolError(f"unknown surface {self.surface!r}")
        if not self.tool_name.strip():
            raise AdmissionProtocolError("tool_name is required")
        body: dict[str, Any] = {
            "surface": self.surface,
            "tool_name": self.tool_name,
            "tool_input": dict(self.tool_input),
        }
        if self.framework_version:
            body["framework_version"] = self.framework_version
        if self.session_id:
            body["session_id"] = self.session_id
        if self.cwd:
            body["cwd"] = self.cwd
        if self.tool_use_id:
            body["tool_use_id"] = self.tool_use_id
        if self.hold_timeout_seconds is not None:
            if self.hold_timeout_seconds < 0:
                raise AdmissionProtocolError("hold_timeout_seconds must not be negative")
            body["hold_timeout_seconds"] = self.hold_timeout_seconds
        return body


@dataclass(frozen=True)
class Decision:
    """One ``POST /v1/admit`` response.

    ``held`` is not an allow: the caller blocks the tool call and either
    waits (``Client.decide``) or gives up. ``updated_input`` is present only on
    ``allow`` and MUST be what runs.
    """

    decision: DecisionKind
    reason: str
    admission_id: str
    risk_score: float = 0.0
    approval_id: str | None = None
    updated_input: dict[str, Any] | None = None
    routed_to: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision == "allow"

    @classmethod
    def from_wire(cls, data: Any) -> Decision:
        """Parse a response body, refusing anything that is not a decision.

        An unrecognized decision value is not a decision. Refusing it here is
        what keeps a future protocol change from being read as an allow.
        """
        if not isinstance(data, Mapping):
            raise AdmissionProtocolError("admission response was not a JSON object")
        decision = data.get("decision")
        if decision not in ("allow", "deny", "held"):
            raise AdmissionProtocolError(
                f"admission returned an unrecognized decision {decision!r}"
            )
        reason = data.get("reason")
        approval_id = data.get("approval_id")
        risk_score = data.get("risk_score", 0.0)
        admission_id = data.get("admission_id")
        updated_input = data.get("updated_input")
        routed_to = data.get("routed_to")
        if updated_input is not None and not isinstance(updated_input, Mapping):
            raise AdmissionProtocolError("updated_input was not an object")
        if not isinstance(risk_score, int | float) or isinstance(risk_score, bool):
            raise AdmissionProtocolError("risk_score was not a number")
        return cls(
            decision=decision,
            reason=reason if isinstance(reason, str) else "",
            admission_id=admission_id if isinstance(admission_id, str) else "",
            risk_score=float(risk_score),
            approval_id=approval_id if isinstance(approval_id, str) and approval_id else None,
            updated_input=dict(updated_input) if updated_input is not None else None,
            routed_to=routed_to if isinstance(routed_to, str) and routed_to else None,
        )


@dataclass(frozen=True)
class Verdict:
    """The outcome of the two-phase decision (``Client.decide``). Never an exception.

    ``allowed`` is the only field a framework integration needs to branch on.
    ``reason`` is the message for the model, phrased so an outage is
    distinguishable from a policy decision. ``error`` is set when the deny is
    the adapter's own (the edge could not be reached or did not answer in
    time); it is ``None`` for a policy deny.
    """

    allowed: bool
    reason: str
    decision: Decision | None = None
    error: AdmissionError | None = None
    updated_input: dict[str, Any] | None = None
    routed_to: str | None = None
    held: bool = False

    @property
    def policy_denied(self) -> bool:
        """The edge judged the call and said no (as opposed to not answering)."""
        return not self.allowed and self.error is None and not self.held

    @property
    def unreachable(self) -> bool:
        """The deny is the adapter's own fail-closed decision."""
        return self.error is not None


@dataclass(frozen=True)
class SandboxPosture:
    """The runtime's OS-sandbox configuration, each key ``None`` when not known.

    A ``None`` is evaluated by the control plane as the unsafe default, so
    silence never suppresses a finding.
    """

    enabled: bool | None = None
    allow_unsandboxed_commands: bool | None = None
    fail_if_unavailable: bool | None = None

    def to_wire(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "allow_unsandboxed_commands": self.allow_unsandboxed_commands,
            "fail_if_unavailable": self.fail_if_unavailable,
        }


@dataclass(frozen=True)
class Attestation:
    """A ``RuntimeAttestation``: the runtime's self-report of its governance posture.

    Self-asserted, and honest by construction: the lists come from the
    configuration the adapter installed, not from what it intended.
    """

    framework: str
    adapter: str
    adapter_version: str
    framework_version: str | None = None
    hooked_tools: tuple[str, ...] = ()
    unhooked_exec_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    other_mcp_servers: tuple[str, ...] = ()
    permission_mode: str | None = None
    sandbox: SandboxPosture | None = None
    config_hash: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_wire(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "framework": self.framework,
            "adapter": self.adapter,
            "adapter_version": self.adapter_version,
        }
        if self.framework_version:
            body["framework_version"] = self.framework_version
        for key, value in (
            ("hooked_tools", self.hooked_tools),
            ("unhooked_exec_tools", self.unhooked_exec_tools),
            ("disallowed_tools", self.disallowed_tools),
            ("other_mcp_servers", self.other_mcp_servers),
        ):
            if value:
                body[key] = list(value)
        if self.permission_mode:
            body["permission_mode"] = self.permission_mode
        if self.sandbox is not None:
            body["sandbox"] = self.sandbox.to_wire()
        if self.config_hash:
            body["config_hash"] = self.config_hash
        body.update(self.extra)
        return body
