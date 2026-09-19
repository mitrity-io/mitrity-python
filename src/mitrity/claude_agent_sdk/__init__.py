"""Claude Agent SDK integration: ``governed_options()``.

Requires the ``claude-agent-sdk`` extra: ``pip install "mitrity[claude-agent-sdk]"``.
"""

from ._governor import (
    DEFAULT_GATEWAY_NAME,
    FRAMEWORK,
    FRAMEWORK_VERSION,
    SURFACE,
    Governor,
    GovernorStats,
    governed_options,
)

__all__ = [
    "DEFAULT_GATEWAY_NAME",
    "FRAMEWORK",
    "FRAMEWORK_VERSION",
    "SURFACE",
    "Governor",
    "GovernorStats",
    "governed_options",
]
