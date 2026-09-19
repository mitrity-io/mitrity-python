# mitrity

MITRITY governance adapter for Python agents: Claude Agent SDK, LangChain,
OpenAI Agents SDK and CrewAI integrations for the MITRITY edge (admission API
and gateway).

The MITRITY gateway governs what an agent asks it to do over MCP. It cannot
see what the agent's *framework* does on its own — the Agent SDK's `Bash`,
`Write`, `Edit` and `WebFetch`, a LangChain `ShellTool`. This package closes
that gap: every such call is admitted by the co-located edge **before it
runs**, with the same policy rules, command analysis, DLP and holds an MCP call
gets, and the same audit trail (`surface=agent_hook`). If the edge cannot be
reached, the call is denied — there is no fail-open mode.

Contract: [iag-specs/sentinel/adapters.md](https://github.com/mitrity-io/iag-specs/blob/main/sentinel/adapters.md)
(wire protocol: [sentinel/admission-api.md](https://github.com/mitrity-io/iag-specs/blob/main/sentinel/admission-api.md)).

## Install

Until the first PyPI release, install from git at a pinned ref (a release tag
once one exists, a commit SHA until then):

```bash
pip install "mitrity[claude-agent-sdk] @ git+https://github.com/mitrity-io/mitrity-python.git@<ref>"
pip install "mitrity[langchain] @ git+https://github.com/mitrity-io/mitrity-python.git@<ref>"
pip install "mitrity[openai-agents] @ git+https://github.com/mitrity-io/mitrity-python.git@<ref>"
pip install "mitrity[crewai] @ git+https://github.com/mitrity-io/mitrity-python.git@<ref>"
```

Python ≥ 3.10. The only runtime dependency is `httpx`; the framework extras
pull in `claude-agent-sdk`, `langchain-core`, `openai-agents` and `crewai`
respectively.

## Prerequisite: a co-located edge

The adapter talks to a `mitrity-gateway` (or `mitrity-mcp-sidecar`) running
next to the agent with an `admission` block:

```yaml
admission:
  enabled: true
  listen_addr: "unix:/run/mitrity/admission.sock"
  token_file: "/run/mitrity/admission.token"
```

The adapter finds it through the same environment variables `mitrity-hook`
uses, with the same defaults:

| Variable | Default | Meaning |
| --- | --- | --- |
| `MITRITY_ADMISSION_ADDR` | `unix:/run/mitrity/admission.sock` (`127.0.0.1:8777` on Windows) | The edge's admission listener. Must be a Unix socket or loopback; anything else is refused. |
| `MITRITY_ADMISSION_TOKEN_FILE` | `/run/mitrity/admission.token` | The per-process token the edge writes at startup. |
| `MITRITY_HOOK_TIMEOUT` | `500ms` (max `30s`) | Deadline for one decision. |
| `MITRITY_HOOK_HOLD_TIMEOUT` | `540s` (max `570s`) | Longest to wait on a human approval; `0` disables waiting. |
| `MITRITY_HOOK_FAIL_MODE` | — | Ignored. Adapters have no fail-open mode. |

## Claude Agent SDK

```python
from claude_agent_sdk import query
from mitrity.claude_agent_sdk import governed_options

options = governed_options(
    gateway={
        "type": "stdio",
        "command": "mitrity-gateway",
        "args": ["--config", "/etc/mitrity/gateway.yaml"],
    },
    allowed_tools=["Bash", "Read", "Write", "mcp__mitrity"],
    system_prompt="You are a careful engineering agent.",
)

async for message in query(prompt="Clean up the build directory", options=options):
    print(message)
```

What `governed_options()` does:

- Installs a `PreToolUse` hook over the execution-capable built-ins (`Bash`,
  `Write`, `Edit`, `MultiEdit`, `NotebookEdit`, `WebFetch`, `WebSearch`) that
  asks the edge before each call. An allow is silent, so your own
  `allowed_tools` / `can_use_tool` flow still applies on top. A deny reaches
  the model as `permissionDecision: "deny"` with the policy reason. A hold
  waits for a human (up to `MITRITY_HOOK_HOLD_TIMEOUT`) and is a deny if nobody
  approves.
- Pins `mcp_servers` to the gateway entry (plus anything you add, which is
  reported as ungoverned), sets `strict_mcp_config=True` and
  `setting_sources=[]` unless you override them.
- Attests the runtime's posture (`POST /v1/attest`) on the first prompt of
  each session: which tools are hooked, which are not, other MCP servers,
  permission mode, sandbox settings. The Python SDK has no `SessionStart`
  callback hook, so the first `UserPromptSubmit` is the trigger.
- Keeps every other option you pass. Your own hooks run after MITRITY's; a
  deny from any hook wins.

For statistics or the attestation object, build the `Governor` yourself:

```python
from mitrity.claude_agent_sdk import Governor

governor = Governor(gateway=gateway_config)
options = governor.options(allowed_tools=["Bash"])
...
print(governor.stats)  # admitted / allowed / denied / held / unreachable / routed
print(governor.attestation())  # what the control plane was told
```

### Routed Bash (governed shell)

When the agent's policy sets `builtin_exec_routing: governed_shell`, an
allowed `Bash` comes back with `updated_input` rewriting the command to
`mitrity-hook exec <ticket>`. The adapter emits it as `updatedInput` with an
explicit allow, and the framework runs the relay instead of the model's
command; the gateway executes the judged bytes in its own sandbox. This needs
the `mitrity-hook` binary on `PATH` and a gateway with `exec.enabled: true`.

### Two approval records per hold

The adapter asks twice for a held action: once without waiting (so an
unreachable edge is noticed in milliseconds), then again with the hold budget.
The edge creates an approval for each; resolve whichever is pending. The
stale one times out on its own. This matches `mitrity-hook`.

## LangChain

```python
from langchain_community.tools import ShellTool
from mitrity.langchain import govern_tools

tools = govern_tools([ShellTool(), my_write_tool], session_id=thread_id)
```

`govern(tool)` returns a `BaseTool` with the same name, description and
schema. Its `_run` / `_arun` admit the call first; a deny raises
`ToolException` with the edge's reason (set `handle_tool_error=True` on the
tool if you want the reason fed back to the model as an error result rather
than raised), and an `updated_input` from the edge replaces the arguments
before the wrapped tool runs.

Two honest limits:

- **Coverage is what you hand it.** A tool that did not go through `govern`
  is invisible to the adapter and to the attestation. `govern_tools` is the
  boundary.
- **Argument names stay yours.** The edge parses commands from the keys
  `command`, `cmd`, `script`, `args` and `argv`. LangChain's `ShellTool`
  calls its argument `commands`, which today is DLP- and content-scanned but
  not parsed as a command tree. The adapter does not rename it; that key
  joins the edge's command family on the edge side.

The session id is, in order: the `session_id` you pass, the
`configurable.thread_id` of the run config, or one id per process.

## OpenAI Agents SDK

```python
from agents import Agent, Runner, ShellTool, function_tool
from mitrity.openai_agents import govern


@function_tool
def run_command(command: str) -> str:
    """Run a shell command on the build host and return its output."""
    ...


agent = govern(Agent(name="ops", tools=[run_command, ShellTool(executor=my_executor)]))
result = await Runner.run(agent, "clean up the build directory")
```

`govern(agent)` returns a clone of the agent whose tools ask the edge before
they run; `govern_tools(tools)` does the same for a list. Each tool type is
blocked the way the SDK itself blocks that type:

- A **function tool** gets a *tool input guardrail*: a deny is
  `reject_content` with the policy reason, which the SDK hands to the model
  as the tool's output without running the tool. An allow is silent. An
  `updated_input` from the edge is merged into the JSON arguments the
  function receives.
- **`ShellTool`** and **`ApplyPatchTool`** go through the SDK's approval
  flow: the adapter's `needs_approval` asks the edge, and on a deny its
  `on_approval` rejects the call with the reason. Your own `needs_approval`
  and `on_approval` still apply when MITRITY allows. The shell executor runs
  the edge's rewritten `commands` when it sends them; an `apply_patch`
  rewrite is a deny (the editor has no call id to apply it to).
- **`LocalShellTool`** (deprecated upstream) has neither a guardrail nor an
  approval, so a deny is returned as the executor's output with nothing run.
- **Hosted tools** (web search, file search, code interpreter, image
  generation, hosted MCP) run in OpenAI's cloud and `ComputerTool` drives a
  computer the SDK does not judge: they pass through untouched and are
  reported as ungoverned in the attestation.

MCP servers on the agent are not touched. Pass the MITRITY gateway server as
`gateway=` so it is not reported as ungoverned; every other server is. The
session id is, in order: the `session_id` you pass, `RunConfig.group_id`, or
one id per process. Handoff targets given as `Agent` objects are governed
too; a `Handoff` object is left as is.

## CrewAI

```python
from crewai import Agent
from mitrity.crewai import govern_tools

agent = Agent(role="ops", goal="...", backstory="...", tools=govern_tools([terminal, writer]))
```

`govern(tool)` returns a `BaseTool` with the same name, description and
schema whose `_run` / `_arun` admit the call first. A deny is returned as a
`ToolFailure` carrying the policy reason — CrewAI's declared channel for a
tool that did not do what it was asked: the agent sees the reason, the
failure is recorded on the task output, and the agent's `tool_failure_policy`
decides whether to continue or abort. An `updated_input` from the edge
replaces the arguments before the wrapped tool runs.

Coverage is what you hand it. Tools CrewAI adds to an agent on its own — the
delegation tools, the code interpreter behind `allow_code_execution=True`,
MCP tools from `mcps` — never pass through `govern` and are invisible to the
adapter; on Linux hosts the MITRITY observer reports their executions after
the fact.

## The wire client

Both integrations sit on `mitrity.admission.Client`, which you can use for
any framework:

```python
from mitrity.admission import AdmitRequest, Client

client = Client()  # discovers the edge from the environment
verdict = client.decide(
    AdmitRequest(surface="custom", tool_name="Bash", tool_input={"command": "rm -rf build"})
)
if not verdict.allowed:
    raise RuntimeError(verdict.reason)
```

`decide()` never raises: it returns a `Verdict` whose `reason` is safe to
show the model and whose `error` is set when the deny is the adapter's own
(the edge could not be reached) rather than a policy decision. `admit()` is
the single round trip and raises an `AdmissionError` subclass on any failure.
Async variants: `decide_async`, `admit_async`, `attest_async`.

## What is not governed

- Tools outside the hook matcher, or LangChain / CrewAI tools not passed to
  `govern`. The attestation names them; the MITRITY dashboard shows the gap.
- Hosted tools of the OpenAI Agents SDK (they run in the vendor's cloud) and
  its `ComputerTool`; reported as ungoverned in the attestation.
- The agent's own code: `subprocess.run` in your application never passes a
  tool boundary. On Linux hosts the MITRITY observer reports it after the
  fact.
- Removing the adapter. It is your code; the control plane detects the
  missing attestation and the silent admission counters, it cannot prevent
  the edit.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
ruff check . && ruff format --check . && mypy --strict && pytest
```

Tests run against an in-process fake edge over a Unix socket; no MITRITY
account and no network are needed. The conformance tests are numbered after
the contract (`C1`–`C20`).

## Security

See [SECURITY.md](SECURITY.md). Report vulnerabilities to soc@mitrity.com.

## License

Apache-2.0. Copyright 2026 MITRITY AB.
