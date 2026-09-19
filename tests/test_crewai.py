"""Conformance C26–C28 for the CrewAI integration."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from crewai.tools import BaseTool, tool
from crewai.tools.tool_failure import ToolFailure
from pydantic import BaseModel, Field

from mitrity.admission import Client
from mitrity.crewai import CrewAIGovernor, GovernedTool, govern, govern_tools
from tests.fake_edge import FakeEdge, allow, deny


class ShellArgs(BaseModel):
    command: str = Field(description="The command")


class Terminal(BaseTool):
    name: str = "terminal"
    description: str = "Run a command"
    args_schema: type[BaseModel] = ShellArgs

    def _run(self, command: str) -> str:
        return f"ran {command}"


class AsyncTerminal(Terminal):
    async def _arun(self, command: str) -> str:
        return f"async ran {command}"


def test_c26_run_admits_first(edge: FakeEdge, client: Client) -> None:
    governed = govern(Terminal(), client=client, session_id="crew-1")
    assert isinstance(governed, GovernedTool) and isinstance(governed, BaseTool)
    assert governed.name == "terminal" and governed.description == Terminal().description
    assert governed.args_schema is ShellArgs
    assert governed.run(command="ls") == "ran ls"
    request = edge.admits()[0].body
    assert request["surface"] == "crewai"
    assert request["tool_name"] == "terminal"
    assert request["tool_input"] == {"command": "ls"}
    assert request["session_id"] == "crew-1"
    assert request["cwd"]
    assert request["framework_version"]
    assert "tool_use_id" not in request
    attestation = edge.attests()[0].body
    assert attestation["framework"] == "crewai"
    assert attestation["adapter"] == "mitrity-python"
    assert attestation["hooked_tools"] == ["terminal"]
    assert "unhooked_exec_tools" not in attestation


def test_c26_structured_tool_path_admits(edge: FakeEdge, client: Client) -> None:
    # The path CrewAI's agents take: to_structured_tool() binds _run.
    structured = govern(Terminal(), client=client).to_structured_tool()
    assert structured.invoke(input={"command": "pwd"}) == "ran pwd"
    assert edge.admits()[0].body["tool_input"] == {"command": "pwd"}


def test_c26_deny_is_a_tool_failure_with_the_reason(edge: FakeEdge, client: Client) -> None:
    governed = govern(Terminal(), client=client)
    edge.script(deny())
    result = governed.run(command="rm -rf /")
    assert isinstance(result, ToolFailure)
    assert "no destructive commands" in result.message
    assert result.as_agent_message() == result.message


def test_c26_unreachable_edge_is_a_tool_failure(edge: FakeEdge, tmp_path: Any) -> None:
    client = Client(
        addr=f"unix:{tmp_path}/nothing.sock", token_file=str(edge.token_file), timeout=0.3
    )
    result = govern(Terminal(), client=client).run(command="ls")
    assert isinstance(result, ToolFailure) and "not a policy decision" in result.message


@pytest.mark.anyio
async def test_c26_arun_admits_before_arun(edge: FakeEdge, client: Client) -> None:
    governed = govern(AsyncTerminal(), client=client)
    assert await governed.arun(command="ls") == "async ran ls"
    assert edge.admits()[0].body["tool_input"] == {"command": "ls"}
    edge.script(deny("async no"))
    result = await governed.arun(command="rm")
    assert isinstance(result, ToolFailure) and "async no" in result.message


@pytest.mark.anyio
async def test_arun_of_a_sync_tool_runs_in_a_thread(edge: FakeEdge, client: Client) -> None:
    governed = govern(Terminal(), client=client)
    assert await governed.arun(command="ls") == "ran ls"


def test_c27_updated_input_runs_instead_of_the_original(edge: FakeEdge, client: Client) -> None:
    governed = govern(Terminal(), client=client)
    edge.script(
        allow(updated_input={"command": "mitrity-hook exec met_9"}, routed_to="governed_shell")
    )
    assert governed.run(command="rm -rf build") == "ran mitrity-hook exec met_9"


def test_c27_positional_argument_and_its_rewrite(edge: FakeEdge, client: Client) -> None:
    governed = govern(Terminal(), client=client)
    assert governed.run("hi") == "ran hi"
    assert edge.admits()[0].body["tool_input"] == {"command": "hi"}
    edge.script(allow(updated_input={"command": "redacted"}))
    assert governed.run("hi") == "ran redacted"
    edge.script(allow(updated_input={"other": "x"}))
    result = governed.run("hi")
    assert isinstance(result, ToolFailure) and "cannot apply" in result.message


def test_c27_decorated_tools_are_governed(edge: FakeEdge, client: Client) -> None:
    @tool("upper")
    def upper(text: str) -> str:
        """Upper-case text."""
        return text.upper()

    governed = govern(upper, client=client)
    assert governed.run(text="hi") == "HI"
    assert edge.admits()[0].body == {
        **edge.admits()[0].body,
        "tool_name": "upper",
        "tool_input": {"text": "hi"},
    }
    assert asyncio.run(governed.arun(text="yo")) == "YO"


def test_c28_govern_tools_shares_one_governor_and_attests_every_name(
    edge: FakeEdge, client: Client
) -> None:
    class Reader(BaseTool):
        name: str = "read_file"
        description: str = "Read"

        def _run(self, path: str) -> str:
            return f"read {path}"

    governed = govern_tools([Terminal(), Reader()], client=client, session_id="s")
    assert [t.name for t in governed] == ["terminal", "read_file"]
    assert isinstance(governed[0], GovernedTool) and isinstance(governed[1], GovernedTool)
    assert governed[0].governor is governed[1].governor
    governed[1].run(path="/x")
    assert edge.attests()[0].body["hooked_tools"] == ["read_file", "terminal"]
    governed[0].run(command="ls")
    assert len(edge.attests()) == 1


def test_c28_a_tool_governed_later_reattests(edge: FakeEdge, client: Client) -> None:
    shared = CrewAIGovernor(client=client, session_id="s")
    first = govern(Terminal(), governor=shared)
    first.run(command="ls")
    assert len(edge.attests()) == 1

    class Writer(BaseTool):
        name: str = "write_file"
        description: str = "Write"

        def _run(self, path: str) -> str:
            return path

    later = govern(Writer(), governor=shared)
    later.run(path="/y")
    assert len(edge.attests()) == 2
    assert edge.attests()[1].body["hooked_tools"] == ["terminal", "write_file"]
    assert edge.attests()[0].body["config_hash"] != edge.attests()[1].body["config_hash"]


def test_usage_limit_is_the_frameworks(edge: FakeEdge, client: Client) -> None:
    governed = govern(Terminal(max_usage_count=1), client=client)
    assert governed.run(command="ls") == "ran ls"
    result = governed.run(command="ls")
    assert isinstance(result, ToolFailure) and "usage limit" in result.message
    assert len(edge.admits()) == 1, "a call the framework refused is never admitted"


def always_cache(_args: Any = None, _result: Any = None) -> bool:
    """A ``cache_function`` that says yes to everything: what a governed tool must not keep."""
    return True


def test_a_governed_tool_never_caches(client: Client) -> None:
    inner = Terminal()
    assert inner.cache_function({"command": "ls"}, "ran ls") is True, "CrewAI caches by default"
    governed = govern(inner, client=client)
    assert governed.cache_function({"command": "ls"}, "ran ls") is False
    # The structured tool CrewAI's agents call carries the same answer.
    structured = governed.to_structured_tool()
    assert structured.cache_function({"command": "ls"}, "ran ls") is False
    # Even when a caching function is handed to the constructor directly.
    direct = GovernedTool(
        inner=inner,
        governor=CrewAIGovernor(client=client),
        name="terminal",
        description="Run a command",
        cache_function=always_cache,
    )
    assert direct.cache_function({"command": "ls"}, "ran ls") is False


def test_c27_a_multi_positional_rewrite_covers_every_argument_or_none(
    edge: FakeEdge, client: Client
) -> None:
    class Pair(BaseTool):
        name: str = "pair"
        description: str = "Two positionals"

        def _run(self, left: str, right: str) -> str:
            return f"{left}+{right}"

    governed = govern(Pair(), client=client)
    assert governed.run("a", "b") == "a+b"
    assert edge.admits()[0].body["tool_input"] == {"input": ["a", "b"]}
    edge.script(allow(updated_input={"input": ["x", "y"]}))
    assert governed.run("a", "b") == "x+y"
    for partial in ({"input": ["x"]}, {"input": "x"}, {"input": ["x", "y", "z"]}):
        edge.script(allow(updated_input=partial))
        result = governed.run("a", "b")
        assert isinstance(result, ToolFailure) and "without covering" in result.message
