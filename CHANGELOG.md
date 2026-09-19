# Changelog

All notable changes to `mitrity` are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the version numbers
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The release
workflow takes a version's section as the notes of its GitHub Release, so every
tag needs a dated `## [X.Y.Z] - YYYY-MM-DD` section (see RELEASING.md).

## [Unreleased]

## [0.1.0] - unreleased

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
- Conformance tests `C1`–`C20` against an in-process fake edge.

[Unreleased]: https://github.com/mitrity-io/mitrity-python/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mitrity-io/mitrity-python/releases/tag/v0.1.0
