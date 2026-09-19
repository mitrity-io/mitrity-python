"""Conformance C13, C18, C20 for the LangChain integration."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import BaseTool, StructuredTool, Tool, ToolException

from mitrity.admission import Client
from mitrity.langchain import GovernedTool, LangChainGovernor, govern, govern_tools
from tests.fake_edge import FakeEdge, allow, deny


def shell(command: str) -> str:
    return f"ran {command}"


async def ashell(command: str) -> str:
    return f"async ran {command}"


def terminal(**kwargs: Any) -> StructuredTool:
    return StructuredTool.from_function(
        shell, coroutine=ashell, name="terminal", description="Run a command", **kwargs
    )


def test_c20_sync_run_admits_first(edge: FakeEdge, client: Client) -> None:
    tool = govern(terminal(), client=client, session_id="thread-1")
    assert isinstance(tool, GovernedTool)
    assert tool.name == "terminal" and tool.description == "Run a command"
    assert tool.args == terminal().args
    assert tool.invoke({"command": "ls"}) == "ran ls"
    request = edge.admits()[0].body
    assert request["surface"] == "langchain"
    assert request["tool_name"] == "terminal"
    assert request["tool_input"] == {"command": "ls"}
    assert request["session_id"] == "thread-1"
    assert request["cwd"]
    assert request["framework_version"]
    attestation = edge.attests()[0].body
    assert attestation["framework"] == "langchain"
    assert attestation["adapter"] == "mitrity-python"
    assert attestation["hooked_tools"] == ["terminal"]
    assert "unhooked_exec_tools" not in attestation


@pytest.mark.anyio
async def test_c20_async_run_admits_first(edge: FakeEdge, client: Client) -> None:
    tool = govern(terminal(), client=client, session_id="thread-1")
    assert await tool.ainvoke({"command": "ls"}) == "async ran ls"
    assert edge.admits()[0].body["tool_input"] == {"command": "ls"}
    assert len(edge.attests()) == 1


@pytest.mark.anyio
async def test_async_falls_back_to_the_sync_implementation(edge: FakeEdge, client: Client) -> None:
    sync_only = StructuredTool.from_function(shell, name="terminal", description="Run a command")
    tool = govern(sync_only, client=client)
    assert await tool.ainvoke({"command": "pwd"}) == "ran pwd"


def test_c20_deny_raises_tool_exception_with_the_reason(edge: FakeEdge, client: Client) -> None:
    tool = govern(terminal(), client=client)
    edge.script(deny())
    with pytest.raises(ToolException, match="no destructive commands"):
        tool.invoke({"command": "rm -rf /"})


def test_c20_deny_honors_handle_tool_error(edge: FakeEdge, client: Client) -> None:
    tool = govern(terminal(handle_tool_error=True), client=client)
    edge.script(deny("policy said no"))
    result = tool.invoke({"command": "rm -rf /"})
    assert "policy said no" in str(result)


def test_unreachable_edge_raises_tool_exception(edge: FakeEdge, tmp_path: Any) -> None:
    client = Client(
        addr=f"unix:{tmp_path}/nothing.sock", token_file=str(edge.token_file), timeout=0.3
    )
    tool = govern(terminal(), client=client)
    with pytest.raises(ToolException, match="not a policy decision"):
        tool.invoke({"command": "ls"})


def test_updated_input_runs_instead_of_the_original(edge: FakeEdge, client: Client) -> None:
    tool = govern(terminal(), client=client)
    edge.script(
        allow(updated_input={"command": "mitrity-hook exec met_9"}, routed_to="governed_shell")
    )
    assert tool.invoke({"command": "rm -rf build"}) == "ran mitrity-hook exec met_9"


def test_c13_single_string_tools_are_sent_under_their_declared_argument(
    edge: FakeEdge, client: Client
) -> None:
    upper = Tool(name="upper", func=lambda text: str(text).upper(), description="Upper-case text")
    tool = govern(upper, client=client)
    assert tool.invoke("hi") == "HI"
    assert edge.admits()[0].body["tool_input"] == {"tool_input": "hi"}


def test_updated_input_replaces_a_positional_argument(edge: FakeEdge, client: Client) -> None:
    upper = Tool(name="upper", func=lambda text: str(text).upper(), description="Upper-case text")
    tool = govern(upper, client=client)
    edge.script(allow(updated_input={"tool_input": "redacted"}))
    assert tool.invoke("hi") == "REDACTED"


def test_thread_id_from_config_is_the_session(edge: FakeEdge, client: Client) -> None:
    tool = govern(terminal(), client=client)
    tool.invoke({"command": "ls"}, config={"configurable": {"thread_id": "t-9"}})
    assert edge.admits()[0].body["session_id"] == "t-9"


def test_default_session_is_one_id_per_process(edge: FakeEdge, client: Client) -> None:
    first = govern(terminal(), client=client)
    second = govern(terminal(), client=client)
    first.invoke({"command": "a"})
    second.invoke({"command": "b"})
    sessions = {r.body["session_id"] for r in edge.admits()}
    assert len(sessions) == 1 and next(iter(sessions)).startswith("langchain-")


def test_govern_tools_shares_one_governor_and_attests_every_name(
    edge: FakeEdge, client: Client
) -> None:
    files = StructuredTool.from_function(
        lambda path: f"read {path}", name="read_file", description="Read"
    )
    governed = govern_tools([terminal(), files], client=client, session_id="s")
    assert [t.name for t in governed] == ["terminal", "read_file"]
    assert isinstance(governed[0], GovernedTool) and isinstance(governed[1], GovernedTool)
    assert governed[0].governor is governed[1].governor
    governed[1].invoke({"path": "/x"})
    assert edge.attests()[0].body["hooked_tools"] == ["read_file", "terminal"]
    assert len(edge.attests()) == 1
    governed[0].invoke({"command": "ls"})
    assert len(edge.attests()) == 1


def test_c18_a_tool_added_later_reattests(edge: FakeEdge, client: Client) -> None:
    shared = LangChainGovernor(client=client, session_id="s")
    first = govern(terminal(), governor=shared)
    first.invoke({"command": "ls"})
    assert len(edge.attests()) == 1
    later = govern(
        StructuredTool.from_function(lambda path: path, name="write_file", description="Write"),
        governor=shared,
    )
    later.invoke({"path": "/y"})
    assert len(edge.attests()) == 2
    assert edge.attests()[1].body["hooked_tools"] == ["terminal", "write_file"]
    assert edge.attests()[0].body["config_hash"] != edge.attests()[1].body["config_hash"]


def test_governed_tool_is_a_base_tool_with_the_inner_schema(client: Client) -> None:
    inner = terminal()
    tool = govern(inner, client=client)
    assert isinstance(tool, BaseTool)
    assert tool.tool_call_schema is inner.tool_call_schema
    assert tool.get_input_schema() is inner.get_input_schema()


def test_a_rewrite_the_adapter_cannot_place_is_a_deny(edge: FakeEdge, client: Client) -> None:
    # Judged bytes execute: a positional call whose rewrite names another key must not run
    # the original value under a decision made about different bytes.
    upper = Tool(name="upper", func=lambda text: str(text).upper(), description="Upper-case text")
    tool = govern(upper, client=client)
    edge.script(allow(updated_input={"command": "something else"}))
    with pytest.raises(ToolException, match="cannot apply"):
        tool.invoke("hi")


def test_a_multi_positional_rewrite_covers_every_argument_or_none(
    edge: FakeEdge, client: Client
) -> None:
    pair = Tool(name="pair", func=lambda left, right: f"{left}+{right}", description="Two")
    tool = govern(pair, client=client)
    assert tool._run("a", "b") == "a+b"
    assert edge.admits()[0].body["tool_input"] == {"tool_input": ["a", "b"]}
    edge.script(allow(updated_input={"tool_input": ["x", "y"]}))
    assert tool._run("a", "b") == "x+y"
    for partial in ({"tool_input": ["x"]}, {"tool_input": "x"}):
        edge.script(allow(updated_input=partial))
        with pytest.raises(ToolException, match="without covering"):
            tool._run("a", "b")
