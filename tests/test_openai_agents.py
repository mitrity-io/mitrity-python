"""Conformance C21–C25 for the OpenAI Agents SDK integration."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import Any, TypeVar, cast

import pytest
from agents import (
    Agent,
    ApplyPatchTool,
    CodeInterpreterTool,
    FunctionTool,
    HostedMCPTool,
    LocalShellTool,
    ModelResponse,
    ModelSettings,
    RunConfig,
    RunContextWrapper,
    Runner,
    ShellTool,
    Tool,
    Usage,
    WebSearchTool,
    function_tool,
    handoff,
)
from agents.editor import ApplyPatchOperation
from agents.handoffs import Handoff
from agents.items import ToolApprovalItem, ToolCallOutputItem, TResponseStreamEvent
from agents.mcp import MCPServer
from agents.models.interface import Model
from agents.tool import (
    LocalShellCommandRequest,
    ShellActionRequest,
    ShellCallData,
    ShellCommandRequest,
)
from agents.tool_context import ToolContext
from agents.tool_guardrails import ToolInputGuardrailData
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputText,
)
from openai.types.responses.response_output_item import LocalShellCall, LocalShellCallAction

from mitrity.admission import Client
from mitrity.openai_agents import (
    GUARDRAIL_NAME,
    MitrityDenied,
    OpenAIAgentsGovernor,
    govern,
    govern_tools,
)
from tests.fake_edge import FakeEdge, allow, deny

T = TypeVar("T")

# ----------------------------------------------------------------- helpers


@function_tool
def run_command(command: str) -> str:
    """Run a shell command."""
    return f"ran {command}"


def only(tools: Sequence[Tool], kind: type[T]) -> T:
    (tool,) = tools
    assert isinstance(tool, kind)
    return tool


async def maybe(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def tool_context(
    tool: FunctionTool,
    arguments: dict[str, Any],
    *,
    call_id: str = "call_1",
    group_id: str | None = None,
) -> ToolContext[Any]:
    run_config = RunConfig(group_id=group_id) if group_id else None
    return ToolContext(
        context=None,
        usage=Usage(),
        tool_name=tool.name,
        tool_call_id=call_id,
        tool_arguments=json.dumps(arguments),
        run_config=run_config,
    )


async def run_guardrail(tool: FunctionTool, ctx: ToolContext[Any]) -> Any:
    guardrail = next(g for g in tool.tool_input_guardrails or [] if g.get_name() == GUARDRAIL_NAME)
    return await guardrail.run(ToolInputGuardrailData(context=ctx, agent=Agent(name="a")))


def ctx_wrapper() -> RunContextWrapper[Any]:
    return RunContextWrapper(context=None)


def approval_item(call_id: str, tool_name: str) -> ToolApprovalItem:
    raw = cast("Any", {"call_id": call_id, "type": f"{tool_name}_call"})
    return ToolApprovalItem(agent=Agent(name="a"), raw_item=raw, tool_name=tool_name)


async def needs_approval(tool: ShellTool | ApplyPatchTool, arg: Any, call_id: str) -> bool:
    fn = tool.needs_approval
    assert callable(fn), "govern() installs a needs_approval callable"
    return bool(await maybe(fn(ctx_wrapper(), arg, call_id)))


async def on_approval(tool: ShellTool | ApplyPatchTool, call_id: str) -> dict[str, Any]:
    fn = tool.on_approval
    assert fn is not None, "govern() installs an on_approval handler"
    return dict(await maybe(fn(ctx_wrapper(), approval_item(call_id, tool.name))))


def shell_request(commands: list[str], call_id: str) -> ShellCommandRequest:
    data = ShellCallData(call_id=call_id, action=ShellActionRequest(commands=commands))
    return ShellCommandRequest(ctx_wrapper=ctx_wrapper(), data=data)


async def run_shell(tool: ShellTool, commands: list[str], call_id: str) -> Any:
    assert tool.executor is not None
    return await maybe(tool.executor(shell_request(commands, call_id)))


async def run_local_shell(tool: LocalShellTool, call: LocalShellCall) -> str:
    request = LocalShellCommandRequest(ctx_wrapper=ctx_wrapper(), data=call)
    return str(await maybe(tool.executor(request)))


def function_call(call_id: str, command: str) -> ResponseFunctionToolCall:
    return ResponseFunctionToolCall(
        id=f"fc_{call_id}",
        call_id=call_id,
        name="run_command",
        arguments=json.dumps({"command": command}),
        type="function_call",
        status="completed",
    )


class StubModel(Model):
    """Answers with a function call, then with a message: enough for one governed tool call."""

    def __init__(self, call: ResponseFunctionToolCall) -> None:
        self._call = call
        self.turns = 0

    async def get_response(self, *args: Any, **kwargs: Any) -> ModelResponse:
        self.turns += 1
        if self.turns == 1:
            return ModelResponse(output=[self._call], usage=Usage(), response_id=None)
        message = ResponseOutputMessage(
            id="msg_1",
            content=[ResponseOutputText(text="done", type="output_text", annotations=[])],
            role="assistant",
            status="completed",
            type="message",
        )
        return ModelResponse(output=[message], usage=Usage(), response_id=None)

    def stream_response(self, *args: Any, **kwargs: Any) -> AsyncIterator[TResponseStreamEvent]:
        raise NotImplementedError


async def run_through_runner(agent: Agent[Any], call: ResponseFunctionToolCall) -> list[Any]:
    run_config = RunConfig(model=StubModel(call), tracing_disabled=True)
    result = await Runner.run(agent, "go", run_config=run_config)
    return [item.output for item in result.new_items if isinstance(item, ToolCallOutputItem)]


# ---------------------------------------------------------------- C21 / C22


@pytest.mark.anyio
async def test_c21_guardrail_admits_before_the_invoker(edge: FakeEdge, client: Client) -> None:
    tool = only(govern_tools([run_command], client=client, session_id="s-1"), FunctionTool)
    ctx = tool_context(tool, {"command": "ls"})
    output = await run_guardrail(tool, ctx)
    assert output.behavior["type"] == "allow"
    assert await tool.on_invoke_tool(ctx, json.dumps({"command": "ls"})) == "ran ls"
    request = edge.admits()[0].body
    assert request["surface"] == "openai_agents"
    assert request["tool_name"] == "run_command"
    assert request["tool_input"] == {"command": "ls"}
    assert request["tool_use_id"] == "call_1"
    assert request["session_id"] == "s-1"
    assert request["cwd"]
    assert request["framework_version"]
    assert len(edge.admits()) == 1, "one decision per call: the invoker reuses the guardrail's"
    attestation = edge.attests()[0].body
    assert attestation["framework"] == "openai-agents"
    assert attestation["adapter"] == "mitrity-python"
    assert attestation["hooked_tools"] == ["run_command"]


@pytest.mark.anyio
async def test_c21_deny_is_reject_content_with_the_reason(edge: FakeEdge, client: Client) -> None:
    tool = only(govern_tools([run_command], client=client), FunctionTool)
    edge.script(deny())
    output = await run_guardrail(tool, tool_context(tool, {"command": "rm -rf /"}))
    assert output.behavior["type"] == "reject_content"
    assert "no destructive commands" in output.behavior["message"]


@pytest.mark.anyio
async def test_c21_rejection_reaches_the_model_through_the_runner(
    edge: FakeEdge, client: Client
) -> None:
    agent = govern(Agent(name="a", instructions="do", tools=[run_command]), client=client)
    edge.script(deny("policy said no"))
    outputs = await run_through_runner(agent, function_call("call_9", "rm -rf /"))
    assert len(outputs) == 1
    assert "policy said no" in str(outputs[0])
    assert edge.admits()[0].body["tool_use_id"] == "call_9"


@pytest.mark.anyio
async def test_c21_allow_runs_the_original_through_the_runner(
    edge: FakeEdge, client: Client
) -> None:
    agent = govern(Agent(name="a", instructions="do", tools=[run_command]), client=client)
    outputs = await run_through_runner(agent, function_call("call_2", "ls"))
    assert outputs == ["ran ls"]
    assert len(edge.admits()) == 1


@pytest.mark.anyio
async def test_c22_updated_input_runs_instead_of_the_original(
    edge: FakeEdge, client: Client
) -> None:
    tool = only(govern_tools([run_command], client=client), FunctionTool)
    ctx = tool_context(tool, {"command": "rm -rf build"})
    edge.script(
        allow(updated_input={"command": "mitrity-hook exec met_9"}, routed_to="governed_shell")
    )
    await run_guardrail(tool, ctx)
    result = await tool.on_invoke_tool(ctx, json.dumps({"command": "rm -rf build"}))
    assert result == "ran mitrity-hook exec met_9"


@pytest.mark.anyio
async def test_c22_arguments_are_verbatim(edge: FakeEdge, client: Client) -> None:
    @function_tool
    def shell(commands: list[str]) -> str:
        """Run commands."""
        return ",".join(commands)

    tool = only(govern_tools([shell], client=client), FunctionTool)
    await run_guardrail(tool, tool_context(tool, {"commands": ["ls", "pwd"]}))
    assert edge.admits()[0].body["tool_input"] == {"commands": ["ls", "pwd"]}


@pytest.mark.anyio
async def test_invoker_admits_when_the_guardrail_was_removed(
    edge: FakeEdge, client: Client
) -> None:
    tool = only(govern_tools([run_command], client=client), FunctionTool)
    ctx = tool_context(tool, {"command": "ls"})
    # No guardrail ran (nothing remembered for this call id): the invoker admits itself.
    assert await tool.on_invoke_tool(ctx, json.dumps({"command": "ls"})) == "ran ls"
    assert len(edge.admits()) == 1
    edge.script(deny("nope"))
    with pytest.raises(MitrityDenied, match="nope"):
        await tool.on_invoke_tool(tool_context(tool, {"command": "rm"}, call_id="call_3"), "{}")


@pytest.mark.anyio
async def test_unreachable_edge_denies_with_the_outage_reason(
    edge: FakeEdge, tmp_path: Any
) -> None:
    client = Client(
        addr=f"unix:{tmp_path}/nothing.sock", token_file=str(edge.token_file), timeout=0.3
    )
    tool = only(govern_tools([run_command], client=client), FunctionTool)
    output = await run_guardrail(tool, tool_context(tool, {"command": "ls"}))
    assert output.behavior["type"] == "reject_content"
    assert "not a policy decision" in output.behavior["message"]


def test_govern_is_idempotent(client: Client) -> None:
    governor = OpenAIAgentsGovernor(client=client)
    once = only(govern_tools([run_command], governor=governor), FunctionTool)
    twice = only(govern_tools([once], governor=governor), FunctionTool)
    assert twice is once
    names = [g.get_name() for g in twice.tool_input_guardrails or []]
    assert names.count(GUARDRAIL_NAME) == 1


# ---------------------------------------------------------------------- C23


@pytest.mark.anyio
async def test_c23_shell_deny_is_an_approval_rejection(edge: FakeEdge, client: Client) -> None:
    ran: list[list[str]] = []

    def executor(request: ShellCommandRequest) -> str:
        ran.append(list(request.data.action.commands))
        return "ok"

    tool = only(govern_tools([ShellTool(executor=executor)], client=client), ShellTool)
    edge.script(deny("shell said no"))
    assert await needs_approval(tool, ShellActionRequest(commands=["rm -rf /"]), "sh_1") is True
    decision = await on_approval(tool, "sh_1")
    assert decision["approve"] is False and "shell said no" in decision["reason"]
    assert ran == []
    request = edge.admits()[0].body
    assert request["tool_name"] == "shell"
    assert request["tool_input"] == {"commands": ["rm -rf /"]}
    assert request["tool_use_id"] == "sh_1"


@pytest.mark.anyio
async def test_c23_shell_allow_defers_to_the_developer_and_runs(
    edge: FakeEdge, client: Client
) -> None:
    ran: list[list[str]] = []

    def executor(request: ShellCommandRequest) -> str:
        ran.append(list(request.data.action.commands))
        return "ok"

    tool = only(govern_tools([ShellTool(executor=executor)], client=client), ShellTool)
    assert await needs_approval(tool, ShellActionRequest(commands=["ls"]), "sh_2") is False, (
        "MITRITY allowed; the developer's default (no approval) applies"
    )
    assert await run_shell(tool, ["ls"], "sh_2") == "ok"
    assert ran == [["ls"]]
    assert len(edge.admits()) == 1, "the executor reuses the decision needs_approval made"


@pytest.mark.anyio
async def test_c23_shell_developer_approval_setting_still_applies(
    edge: FakeEdge, client: Client
) -> None:
    inner = ShellTool(executor=lambda request: "ok", needs_approval=True)
    tool = only(govern_tools([inner], client=client), ShellTool)
    assert await needs_approval(tool, ShellActionRequest(commands=["ls"]), "sh_3") is True
    decision = await on_approval(tool, "sh_3")
    assert decision.get("approve") is None, "MITRITY allowed: the developer's interrupt stands"


@pytest.mark.anyio
async def test_c23_shell_executor_runs_rewritten_commands(edge: FakeEdge, client: Client) -> None:
    ran: list[list[str]] = []

    async def executor(request: ShellCommandRequest) -> str:
        ran.append(list(request.data.action.commands))
        return "ok"

    tool = only(govern_tools([ShellTool(executor=executor)], client=client), ShellTool)
    edge.script(
        allow(updated_input={"commands": ["mitrity-hook exec met_1"]}, routed_to="governed_shell")
    )
    action = ShellActionRequest(commands=["rm -rf build"])
    assert await needs_approval(tool, action, "sh_4") is False
    await run_shell(tool, ["rm -rf build"], "sh_4")
    assert ran == [["mitrity-hook exec met_1"]]


@pytest.mark.anyio
async def test_c23_apply_patch_deny_and_rewrite(edge: FakeEdge, client: Client) -> None:
    class Editor:
        def create_file(self, operation: ApplyPatchOperation) -> str:
            return "created"

        def update_file(self, operation: ApplyPatchOperation) -> str:
            return "updated"

        def delete_file(self, operation: ApplyPatchOperation) -> str:
            return "deleted"

    tool = only(govern_tools([ApplyPatchTool(editor=Editor())], client=client), ApplyPatchTool)
    op = ApplyPatchOperation(type="delete_file", path="/etc/passwd")
    edge.script(deny("patch said no"))
    assert await needs_approval(tool, op, "ap_1") is True
    rejected = await on_approval(tool, "ap_1")
    assert rejected["approve"] is False and "patch said no" in rejected["reason"]
    assert edge.admits()[0].body["tool_input"] == {"type": "delete_file", "path": "/etc/passwd"}
    assert edge.admits()[0].body["tool_name"] == "apply_patch"
    # An allow that rewrites is a deny: the editor has no call id to apply it to.
    edge.script(allow(updated_input={"path": "/tmp/elsewhere"}))
    assert await needs_approval(tool, op, "ap_2") is True
    decision = await on_approval(tool, "ap_2")
    assert decision["approve"] is False and "cannot apply" in decision["reason"]
    # A plain allow defers to the developer (default: no approval).
    assert await needs_approval(tool, op, "ap_3") is False


@pytest.mark.anyio
async def test_c23_local_shell_deny_is_the_output(edge: FakeEdge, client: Client) -> None:
    ran: list[list[str]] = []

    def executor(request: LocalShellCommandRequest) -> str:
        ran.append(list(request.data.action.command))
        return "ok"

    tool = only(govern_tools([LocalShellTool(executor=executor)], client=client), LocalShellTool)
    call = LocalShellCall(
        id="lsh_1",
        call_id="lsh_1",
        action=LocalShellCallAction(command=["rm", "-rf", "/"], env={}, type="exec"),
        status="completed",
        type="local_shell_call",
    )
    edge.script(deny("local shell said no"))
    output = await run_local_shell(tool, call)
    assert "local shell said no" in output and ran == []
    assert edge.admits()[0].body["tool_input"]["command"] == ["rm", "-rf", "/"]
    assert await run_local_shell(tool, call) == "ok"
    assert ran == [["rm", "-rf", "/"]]


# ---------------------------------------------------------------- C24 / C25


class NamedServer(MCPServer):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def connect(self) -> None: ...

    async def cleanup(self) -> None: ...

    async def list_tools(self, *args: Any, **kwargs: Any) -> Any:
        return []

    async def call_tool(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def list_prompts(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def get_prompt(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.mark.anyio
async def test_c24_hosted_and_mcp_are_attested(edge: FakeEdge, client: Client) -> None:
    gateway = NamedServer("mitrity")
    other = NamedServer("github")
    interpreter = CodeInterpreterTool(
        tool_config={"type": "code_interpreter", "container": {"type": "auto"}}
    )
    hosted = HostedMCPTool(
        tool_config={"type": "mcp", "server_label": "docs", "server_url": "https://x"}
    )
    agent = Agent(
        name="a",
        tools=[run_command, WebSearchTool(), interpreter, hosted],
        mcp_servers=[gateway, other],
    )
    governed = govern(agent, client=client, gateway=gateway)
    assert governed is not agent and governed.mcp_servers == [gateway, other]
    tool = only([t for t in governed.tools if isinstance(t, FunctionTool)], FunctionTool)
    await run_guardrail(tool, tool_context(tool, {"command": "ls"}))
    attestation = edge.attests()[0].body
    assert attestation["hooked_tools"] == ["run_command"]
    assert attestation["unhooked_exec_tools"] == ["code_interpreter", "hosted_mcp", "web_search"]
    assert attestation["other_mcp_servers"] == ["github", "hosted:docs"]


@pytest.mark.anyio
async def test_c25_session_is_the_run_config_group_id(edge: FakeEdge, client: Client) -> None:
    tool = only(govern_tools([run_command], client=client), FunctionTool)
    await run_guardrail(tool, tool_context(tool, {"command": "ls"}, group_id="conv-7"))
    assert edge.admits()[0].body["session_id"] == "conv-7"
    other = only(govern_tools([run_command], client=client), FunctionTool)
    await run_guardrail(other, tool_context(other, {"command": "ls"}, call_id="call_x"))
    assert edge.admits()[1].body["session_id"].startswith("openai-agents-")


@pytest.mark.anyio
async def test_a_tool_governed_later_reattests(edge: FakeEdge, client: Client) -> None:
    governor = OpenAIAgentsGovernor(client=client, session_id="s")
    first = only(govern_tools([run_command], governor=governor), FunctionTool)
    await run_guardrail(first, tool_context(first, {"command": "ls"}))
    assert len(edge.attests()) == 1

    @function_tool
    def write_file(path: str) -> str:
        """Write."""
        return path

    later = only(govern_tools([write_file], governor=governor), FunctionTool)
    await run_guardrail(later, tool_context(later, {"path": "/y"}, call_id="call_y"))
    assert len(edge.attests()) == 2
    assert edge.attests()[1].body["hooked_tools"] == ["run_command", "write_file"]
    assert edge.attests()[0].body["config_hash"] != edge.attests()[1].body["config_hash"]


def test_handoff_agents_are_governed_with_the_same_governor(client: Client) -> None:
    child = Agent(name="child", tools=[run_command])
    parent = Agent(name="parent", tools=[run_command], handoffs=[child])
    governed = govern(parent, client=client)
    (child_governed,) = governed.handoffs
    assert isinstance(child_governed, Agent)
    tool = only(child_governed.tools, FunctionTool)
    assert any(g.get_name() == GUARDRAIL_NAME for g in tool.tool_input_guardrails or [])


def test_model_settings_untouched(client: Client) -> None:
    agent = Agent(name="a", tools=[run_command], model_settings=ModelSettings(temperature=0.1))
    governed = govern(agent, client=client)
    assert governed.model_settings == agent.model_settings and governed.name == "a"


# -------------------------------------------------------- handoffs and tool types


def governed_names(agent: Agent[Any]) -> list[str]:
    return [
        t.name
        for t in agent.tools
        if isinstance(t, FunctionTool)
        and any(g.get_name() == GUARDRAIL_NAME for g in t.tool_input_guardrails or [])
    ]


def test_govern_terminates_on_a_handoff_cycle(client: Client) -> None:
    triage = Agent(name="triage", tools=[run_command])
    faq = Agent(name="faq", tools=[run_command], handoffs=[triage])
    triage.handoffs.append(faq)
    governed = govern(triage, client=client)
    (faq_governed,) = governed.handoffs
    assert isinstance(faq_governed, Agent) and faq_governed is not faq
    assert governed_names(faq_governed) == ["run_command"]
    (back,) = faq_governed.handoffs
    assert back is governed, "the hand-back resolves to the one governed triage clone"
    assert triage.handoffs == [faq] and faq.handoffs == [triage], "the originals are untouched"


def test_govern_shares_one_clone_across_a_diamond(client: Client) -> None:
    leaf = Agent(name="leaf", tools=[run_command])
    left = Agent(name="left", handoffs=[leaf])
    right = Agent(name="right", handoffs=[leaf])
    root = Agent(name="root", handoffs=[left, right])
    governed = govern(root, client=client)
    left_governed, right_governed = governed.handoffs
    assert isinstance(left_governed, Agent) and isinstance(right_governed, Agent)
    (via_left,) = left_governed.handoffs
    (via_right,) = right_governed.handoffs
    assert via_left is via_right and via_left is not leaf
    assert isinstance(via_left, Agent) and governed_names(via_left) == ["run_command"]


def test_govern_agent_is_idempotent(client: Client) -> None:
    governor = OpenAIAgentsGovernor(client=client)
    once = govern(Agent(name="a", tools=[run_command]), governor=governor)
    assert govern(once, governor=governor) is once
    parent = govern(Agent(name="p", handoffs=[once]), governor=governor)
    assert parent.handoffs == [once]


def test_handoff_objects_are_left_as_is_with_a_warning(
    client: Client, caplog: pytest.LogCaptureFixture
) -> None:
    child = Agent(name="child", tools=[run_command])
    built = handoff(child)
    parent = Agent(name="parent", handoffs=[built])
    with caplog.at_level(logging.WARNING, logger="mitrity.openai_agents"):
        governed = govern(parent, client=client)
    assert governed.handoffs == [built] and isinstance(built, Handoff)
    assert "handoff()" in caplog.text
