"""OpenAI Agents SDK integration: ``govern()`` and ``govern_tools()``.

Requires the ``openai-agents`` extra: ``pip install "mitrity[openai-agents]"``.
"""

from ._governed import (
    FRAMEWORK,
    FRAMEWORK_VERSION,
    GUARDRAIL_NAME,
    SURFACE,
    GovernorStats,
    MitrityDenied,
    OpenAIAgentsGovernor,
    govern,
    govern_tools,
)

__all__ = [
    "FRAMEWORK",
    "FRAMEWORK_VERSION",
    "GUARDRAIL_NAME",
    "SURFACE",
    "GovernorStats",
    "MitrityDenied",
    "OpenAIAgentsGovernor",
    "govern",
    "govern_tools",
]
