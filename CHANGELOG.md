# Changelog

All notable changes to `mitrity` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The release
workflow takes a version's section as the notes of its GitHub Release, so every
tag needs a dated `## [X.Y.Z] - YYYY-MM-DD` section (see RELEASING.md).

## [Unreleased]

### Changed

- The README, `SECURITY.md`, the docstrings and the `Contract` project URL link
  to the public adapter contract and admission API pages
  (https://mitrity.com/docs/integrations/adapters and
  https://mitrity.com/docs/integrations/admission-api) instead of the
  specification repository.

## [0.2.0] - 2026-09-20

The first published version. It carries the four adapters and the wire client.

### Added

- `mitrity.admission`: the wire client for the MITRITY edge's admission API
  (`Client.decide()`, `admit()`, `attest()` and their async variants), with the
  same edge discovery, loopback-only and fail-closed rules as `mitrity-hook`.
- `mitrity.claude_agent_sdk`: `governed_options()` and `Governor` for the Claude
  Agent SDK. A `PreToolUse` hook admits the execution-capable built-in tools
  through the edge before they run, `mcp_servers` is pinned to the MITRITY
  gateway, routed Bash (governed shell) is honored, and the runtime's posture is
  attested on the first prompt of each session.
- `mitrity.langchain`: `govern()` and `govern_tools()` wrap LangChain tools so
  every call is admitted before the tool runs.
- `mitrity.openai_agents`: `govern()` and `govern_tools()` for the OpenAI Agents
  SDK. Function tools get a tool input guardrail and a wrapped invoker that
  applies the edge's `updated_input`; the shell and apply-patch tools go through
  the SDK's approval flow; hosted tools and the computer tool pass through and are
  attested as unhooked; handoff agents are governed with the same governor.
- `mitrity.crewai`: `govern()` and `govern_tools()` for CrewAI. `GovernedTool`
  keeps the inner tool's name, description, schema and policies, admits every
  call first, and returns CrewAI's `ToolFailure` with the edge's reason on a deny.
- Conformance tests `C1`–`C20` against an in-process fake edge.

[Unreleased]: https://github.com/mitrity-io/mitrity-python/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mitrity-io/mitrity-python/releases/tag/v0.2.0
