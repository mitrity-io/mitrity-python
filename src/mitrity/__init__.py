"""MITRITY governance adapter for Python agents.

The package connects an agent framework's tool execution to the co-located
MITRITY edge: built-in tools are admitted through the loopback admission API
before they run, MCP tools reach the model through the MITRITY gateway, and the
runtime's governance posture is attested so the control plane can render an
honest coverage badge.

- :mod:`mitrity.admission` — the wire client (no framework dependency).
- :mod:`mitrity.claude_agent_sdk` — ``governed_options()`` for the Claude Agent SDK.
- :mod:`mitrity.langchain` — ``govern()`` / ``govern_tools()`` for LangChain tools.

Contract: https://github.com/mitrity-io/iag-specs/blob/main/sentinel/adapters.md
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
