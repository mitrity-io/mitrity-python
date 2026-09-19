"""CrewAI integration: ``govern()`` and ``govern_tools()``.

Requires the ``crewai`` extra: ``pip install "mitrity[crewai]"``.
"""

from ._governed import (
    FRAMEWORK,
    FRAMEWORK_VERSION,
    SURFACE,
    CrewAIGovernor,
    GovernedTool,
    govern,
    govern_tools,
)

__all__ = [
    "FRAMEWORK",
    "FRAMEWORK_VERSION",
    "SURFACE",
    "CrewAIGovernor",
    "GovernedTool",
    "govern",
    "govern_tools",
]
