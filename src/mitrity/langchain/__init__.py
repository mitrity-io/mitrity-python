"""LangChain integration: ``govern()`` and ``govern_tools()``.

Requires the ``langchain`` extra: ``pip install "mitrity[langchain]"``.
"""

from ._governed import (
    FRAMEWORK,
    FRAMEWORK_VERSION,
    SURFACE,
    GovernedTool,
    LangChainGovernor,
    govern,
    govern_tools,
)

__all__ = [
    "FRAMEWORK",
    "FRAMEWORK_VERSION",
    "SURFACE",
    "GovernedTool",
    "LangChainGovernor",
    "govern",
    "govern_tools",
]
