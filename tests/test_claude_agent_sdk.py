"""Conformance C7, C8, C10–C13, C16, C18, C20 for the Claude Agent SDK integration.

The hooks are exercised directly with the payloads the SDK delivers; no CLI is
spawned. What the SDK does with a hook output is its contract, documented in
the Claude Code hooks reference; what the adapter puts in it is ours.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from typing import Any, cast

import pytest
from claude_agent_sdk import ClaudeAgentOptions, HookMatcher
from claude_agent_sdk.types import HookContext, HookInput, HookJSONOutput

from mitrity import __version__
from mitrity.admission import EXEC_CAPABLE_TOOLS, AdmissionError, AdmitRequest, Client, Verdict
from mitrity.claude_agent_sdk import FRAMEWORK_VERSION, Governor, governed_options
from tests.fake_edge import FakeEdge, allow, deny, held

GATEWAY: dict[str, Any] = {
    "type": "stdio",
    "command": "mitrity-gateway",
    "args": ["--config", "/etc/mitrity/gateway.yaml"],
}
CONTEXT: HookContext = {"signal": None}


def hook_input(
    tool_name: str,
    tool_input: dict[str, Any] | None = None,
    *,
    event: str = "PreToolUse",
    session_id: str = "sess-1",
    cwd: str = "/workspace/repo",
    tool_use_id: str = "toolu_1",
    permission_mode: str = "default",
) -> HookInput:
    payload: dict[str, Any] = {
        "session_id": session_id,
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": cwd,
        "permission_mode": permission_mode,
        "hook_event_name": event,
        "tool_name": tool_name,
        "tool_input": tool_input if tool_input is not None else {},
        "tool_use_id": tool_use_id,
    }
    if event == "PostToolUse":
        payload["tool_response"] = {"stdout": "", "stderr": ""}
    return cast("HookInput", payload)


def prompt_input(session_id: str = "sess-1", permission_mode: str = "default") -> HookInput:
    return cast(
        "HookInput",
        {
            "session_id": session_id,
            "transcript_path": "/tmp/transcript.jsonl",
            "cwd": "/workspace/repo",
            "permission_mode": permission_mode,
            "hook_event_name": "UserPromptSubmit",
            "prompt": "hello",
        },
    )


def governor(client: Client, **kwargs: Any) -> Governor:
    return Governor(client=client, gateway=cast("Any", GATEWAY), **kwargs)


def test_options_shape(client: Client) -> None:
    options = governed_options(
        client=client, gateway=cast("Any", GATEWAY), permission_mode="default"
    )
    assert isinstance(options, ClaudeAgentOptions)
    assert options.mcp_servers == {"mitrity": GATEWAY}
    assert options.strict_mcp_config is True
    assert options.setting_sources == []
    assert options.permission_mode == "default"
    hooks = options.hooks or {}
    pre = hooks["PreToolUse"][0]
    assert pre.matcher == "|".join(EXEC_CAPABLE_TOOLS)
    # attest 5 s + decision 0.5 s + hold 2 s + margin 5 s + slack 30 s
    assert pre.timeout == 42.5
    assert hooks["PostToolUse"][0].matcher is None
    assert hooks["UserPromptSubmit"][0].matcher is None


def test_developer_hooks_are_kept_after_ours(client: Client) -> None:
    async def theirs(
        input_data: HookInput, tool_use_id: str | None, context: HookContext
    ) -> HookJSONOutput:
        return {}

    options = governed_options(
        client=client,
        gateway=cast("Any", GATEWAY),
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[theirs])]},
    )
    matchers = (options.hooks or {})["PreToolUse"]
    assert len(matchers) == 2
    assert matchers[1].hooks == [theirs]


def test_mcp_servers_as_a_path_is_refused(client: Client) -> None:
    with pytest.raises(TypeError):
        governed_options(client=client, mcp_servers="/etc/mcp.json")


def test_a_governor_builds_one_options_object(client: Client) -> None:
    gov = governor(client)
    gov.options()
    with pytest.raises(RuntimeError):
        gov.options()
    with pytest.raises(RuntimeError):
        Governor(client=client).attestation()


@pytest.mark.anyio
async def test_c7_c13_allow_is_silent_and_the_request_is_verbatim(
    edge: FakeEdge, client: Client
) -> None:
    gov = governor(client)
    gov.options()
    out = await gov.pre_tool_use(
        hook_input("Bash", {"command": "ls", "description": "list", "timeout": 5000}),
        "toolu_1",
        CONTEXT,
    )
    assert out == {}
    request = edge.admits()[0].body
    assert request == {
        "surface": "claude_agent_sdk",
        "framework_version": FRAMEWORK_VERSION,
        "session_id": "sess-1",
        "cwd": "/workspace/repo",
        "tool_name": "Bash",
        "tool_input": {"command": "ls", "description": "list", "timeout": 5000},
        "tool_use_id": "toolu_1",
        "hold_timeout_seconds": 0,
    }
    assert gov.stats.admitted == 1 and gov.stats.allowed == 1


@pytest.mark.anyio
async def test_c13_argument_names_are_not_renamed(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()
    await gov.pre_tool_use(hook_input("Bash", {"commands": ["ls"]}), "toolu_1", CONTEXT)
    assert edge.admits()[0].body["tool_input"] == {"commands": ["ls"]}


@pytest.mark.anyio
async def test_c8_deny_in_the_framework_idiom(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()
    edge.script(deny())
    out = await gov.pre_tool_use(hook_input("Bash", {"command": "rm -rf /"}), "toolu_1", CONTEXT)
    specific = cast("dict[str, Any]", out)["hookSpecificOutput"]
    assert specific["hookEventName"] == "PreToolUse"
    assert specific["permissionDecision"] == "deny"
    assert specific["permissionDecisionReason"] == (
        'MITRITY denied this action: policy rule "no destructive commands" '
        "denied resolved command: rm"
    )
    assert gov.stats.denied == 1


@pytest.mark.anyio
async def test_c10_routed_allow_emits_merged_updated_input(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()
    edge.script(
        allow(updated_input={"command": "mitrity-hook exec met_1"}, routed_to="governed_shell")
    )
    out = await gov.pre_tool_use(
        hook_input("Bash", {"command": "rm -rf build", "description": "clean"}), "toolu_1", CONTEXT
    )
    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {"command": "mitrity-hook exec met_1", "description": "clean"},
        }
    }
    assert gov.stats.routed == 1


@pytest.mark.anyio
async def test_held_after_budget_is_a_deny_with_a_system_message(
    edge: FakeEdge, client: Client
) -> None:
    edge.default_admit = held("apr-9")
    gov = governor(client)
    gov.options()
    out = cast(
        "dict[str, Any]", await gov.pre_tool_use(hook_input("Bash", {"command": "x"}), "t", CONTEXT)
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "apr-9" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert "human approval" in out["systemMessage"]
    assert gov.stats.held == 1


@pytest.mark.anyio
async def test_unreachable_edge_is_a_deny_that_says_so(edge: FakeEdge, tmp_path: Path) -> None:
    client = Client(
        addr=f"unix:{tmp_path}/nothing.sock", token_file=str(edge.token_file), timeout=0.3
    )
    gov = governor(client)
    gov.options()
    out = cast(
        "dict[str, Any]", await gov.pre_tool_use(hook_input("Bash", {"command": "x"}), "t", CONTEXT)
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "not a policy decision" in out["hookSpecificOutput"]["permissionDecisionReason"]
    assert gov.stats.unreachable == 1


@pytest.mark.anyio
async def test_adapter_failure_is_a_deny_not_an_exception(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()

    async def explode(request: AdmitRequest, **kwargs: Any) -> Verdict:
        raise RuntimeError("boom")

    cast("Any", client).decide_async = explode
    out = cast(
        "dict[str, Any]", await gov.pre_tool_use(hook_input("Bash", {"command": "x"}), "t", CONTEXT)
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert "boom" in out["hookSpecificOutput"]["permissionDecisionReason"]


@pytest.mark.anyio
async def test_missing_tool_name_is_a_deny(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()
    out = cast(
        "dict[str, Any]", await gov.pre_tool_use(hook_input("", {"command": "x"}), "t", CONTEXT)
    )
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert edge.admits() == []


@pytest.mark.anyio
async def test_c20_mcp_tools_are_never_admitted_here(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()
    out = await gov.pre_tool_use(
        hook_input("mcp__mitrity__fs:read_file", {"path": "/x"}), "t", CONTEXT
    )
    assert out == {}
    assert edge.admits() == []


@pytest.mark.anyio
async def test_c11_attests_once_per_session_on_the_first_prompt(
    edge: FakeEdge, client: Client
) -> None:
    gov = governor(client)
    gov.options(sandbox={"enabled": True, "allowUnsandboxedCommands": False})
    await gov.user_prompt_submit(prompt_input(), None, CONTEXT)
    await gov.user_prompt_submit(prompt_input(), None, CONTEXT)
    await gov.pre_tool_use(hook_input("Bash", {"command": "ls"}), "t", CONTEXT)
    assert len(edge.attests()) == 1
    body = edge.attests()[0].body
    assert body["framework"] == "claude-agent-sdk"
    assert body["framework_version"] == FRAMEWORK_VERSION
    assert body["adapter"] == "mitrity-python"
    assert body["adapter_version"] == __version__
    assert body["hooked_tools"] == sorted(EXEC_CAPABLE_TOOLS)
    assert "unhooked_exec_tools" not in body
    assert "other_mcp_servers" not in body
    assert body["permission_mode"] == "default"
    assert body["sandbox"] == {
        "enabled": True,
        "allow_unsandboxed_commands": False,
        "fail_if_unavailable": None,
    }
    assert re.fullmatch(r"[0-9a-f]{64}", body["config_hash"])
    assert gov.stats.attestations == 1
    await gov.user_prompt_submit(prompt_input(session_id="sess-2"), None, CONTEXT)
    assert len(edge.attests()) == 2


def test_c12_coverage_lists_are_honest(client: Client) -> None:
    gov = governor(client, hooked_tools=["Bash", "Write"])
    options = gov.options(
        disallowed_tools=["WebFetch"],
        mcp_servers={"other": {"type": "http", "url": "https://example.com/mcp"}},
        strict_mcp_config=False,
        setting_sources=["user", "project"],
        sandbox={"enabled": True, "allowUnsandboxedCommands": False},
    )
    assert isinstance(options.mcp_servers, dict)
    assert set(options.mcp_servers) == {"mitrity", "other"}
    attestation = gov.attestation()
    assert attestation.hooked_tools == ("Bash", "Write")
    # Nothing is subtracted here: the control plane subtracts disallowed_tools,
    # exactly as it does for the hook's attestation.
    assert attestation.unhooked_exec_tools == (
        "Edit",
        "MultiEdit",
        "NotebookEdit",
        "WebFetch",
        "WebSearch",
    )
    assert attestation.disallowed_tools == ("WebFetch",)
    assert attestation.other_mcp_servers == ("other", "settings:project", "settings:user")
    assert attestation.sandbox is not None
    assert attestation.sandbox.enabled is True
    assert attestation.sandbox.allow_unsandboxed_commands is False
    assert attestation.sandbox.fail_if_unavailable is None


def test_c12_non_strict_with_default_sources_names_every_source(client: Client) -> None:
    gov = governor(client)
    gov.options(strict_mcp_config=False, setting_sources=None)
    assert gov.attestation().other_mcp_servers == (
        "settings:local",
        "settings:project",
        "settings:user",
    )


def test_c12_tools_option_removes_tools_from_both_lists(client: Client) -> None:
    gov = governor(client)
    options = gov.options(tools=["Bash", "Read"])
    assert (options.hooks or {})["PreToolUse"][0].matcher == "Bash"
    attestation = gov.attestation()
    assert attestation.hooked_tools == ("Bash",)
    assert attestation.unhooked_exec_tools == ()


def test_no_gateway_means_every_server_is_other(client: Client) -> None:
    gov = Governor(client=client)
    options = gov.options(mcp_servers={"tools": {"type": "http", "url": "https://example.com"}})
    assert options.mcp_servers == {"tools": {"type": "http", "url": "https://example.com"}}
    assert gov.attestation().other_mcp_servers == ("tools",)


@pytest.mark.anyio
async def test_c16_concurrent_calls_are_independent(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()
    outputs = await asyncio.gather(
        *(
            gov.pre_tool_use(
                hook_input("Bash", {"command": f"echo {i}"}, tool_use_id=f"toolu_{i}"),
                f"toolu_{i}",
                CONTEXT,
            )
            for i in range(10)
        )
    )
    assert all(out == {} for out in outputs)
    ids = sorted(r.body["tool_use_id"] for r in edge.admits())
    assert ids == sorted(f"toolu_{i}" for i in range(10))


@pytest.mark.anyio
async def test_c18_permission_mode_change_reattests(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()
    await gov.pre_tool_use(hook_input("Bash", {"command": "ls"}), "t1", CONTEXT)
    await gov.pre_tool_use(
        hook_input("Bash", {"command": "ls"}, permission_mode="bypassPermissions"), "t2", CONTEXT
    )
    hashes = [r.body["config_hash"] for r in edge.attests()]
    assert len(hashes) == 2 and hashes[0] != hashes[1]
    assert edge.attests()[1].body["permission_mode"] == "bypassPermissions"


@pytest.mark.anyio
async def test_c18_unadmitted_execution_widens_unhooked_and_reattests(
    edge: FakeEdge, client: Client
) -> None:
    gov = governor(client)
    gov.options(tools=["Bash"])
    await gov.user_prompt_submit(prompt_input(), None, CONTEXT)
    assert "unhooked_exec_tools" not in edge.attests()[0].body
    await gov.post_tool_use(
        hook_input("Write", {"file_path": "/x"}, event="PostToolUse", tool_use_id="never"),
        "never",
        CONTEXT,
    )
    assert len(edge.attests()) == 2
    assert edge.attests()[1].body["unhooked_exec_tools"] == ["Write"]
    assert gov.stats.unadmitted_executions == 1


@pytest.mark.anyio
async def test_post_tool_use_of_an_admitted_call_is_quiet(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()
    await gov.pre_tool_use(hook_input("Bash", {"command": "ls"}), "toolu_1", CONTEXT)
    out = await gov.post_tool_use(
        hook_input("Bash", {"command": "ls"}, event="PostToolUse"), "toolu_1", CONTEXT
    )
    assert out == {}
    assert len(edge.attests()) == 1
    assert gov.stats.unadmitted_executions == 0


@pytest.mark.anyio
async def test_attestation_failure_never_blocks_a_call(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()

    async def refuse(*_: Any, **__: Any) -> None:
        raise AdmissionError("attest down")

    cast("Any", client).attest_async = refuse
    out = await gov.pre_tool_use(hook_input("Bash", {"command": "ls"}), "t", CONTEXT)
    assert out == {}
    assert gov.stats.attestations == 0


def test_hook_budget_fits_the_framework_budget(edge: FakeEdge) -> None:
    # Every term the adapter can spend inside one hook has to fit under 600 s; the hold
    # budget is what gives.
    client = Client(
        addr=edge.addr, token_file=str(edge.token_file), timeout=30.0, hold_timeout=570.0
    )
    gov = governor(client)
    options = gov.options()
    pre = (options.hooks or {})["PreToolUse"][0]
    assert pre.timeout == 600.0
    assert gov.hold_budget == 600.0 - (5.0 + 30.0 + 5.0 + 30.0)
    assert gov.hold_budget < client.config.hold_timeout


@pytest.mark.anyio
async def test_hold_wait_uses_the_fitted_budget(edge: FakeEdge) -> None:
    from tests.fake_edge import HoldScript
    from tests.fake_edge import allow as allow_response

    client = Client(
        addr=edge.addr, token_file=str(edge.token_file), timeout=30.0, hold_timeout=570.0
    )
    edge.default_admit = HoldScript(outcome=allow_response())
    gov = governor(client)
    gov.options()
    out = await gov.pre_tool_use(hook_input("Bash", {"command": "x"}), "t", CONTEXT)
    assert out == {}
    budgets = [r.body["hold_timeout_seconds"] for r in edge.admits()]
    assert budgets == [0, int(gov.hold_budget)]


@pytest.mark.anyio
async def test_adapter_failure_reason_is_bounded(edge: FakeEdge, client: Client) -> None:
    gov = governor(client)
    gov.options()

    async def explode(request: AdmitRequest, **kwargs: Any) -> Verdict:
        raise RuntimeError("x" * 5000)

    cast("Any", client).decide_async = explode
    out = cast(
        "dict[str, Any]", await gov.pre_tool_use(hook_input("Bash", {"command": "x"}), "t", CONTEXT)
    )
    assert len(out["hookSpecificOutput"]["permissionDecisionReason"]) < 400
