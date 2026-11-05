"""Verify tool package.

Provides a ``verify`` tool that runs a project's build/test command and reports
a structured pass/fail result, so an agent can confirm its change works instead
of only reasoning about it.

Usage:
    from openhands.tools.verify import VerifyTool

    agent = Agent(
        llm=llm,
        tools=[Tool(name=VerifyTool.name)],
    )
"""

from openhands.tools.verify.definition import (
    DEFAULT_COMMANDS,
    PROJECT_MARKERS,
    VerifyAction,
    VerifyObservation,
    VerifyTool,
    detect_project_type,
    resolve_command,
)
from openhands.tools.verify.impl import VerifyExecutor


__all__ = [
    "DEFAULT_COMMANDS",
    "PROJECT_MARKERS",
    "VerifyAction",
    "VerifyObservation",
    "VerifyTool",
    "VerifyExecutor",
    "detect_project_type",
    "resolve_command",
]
