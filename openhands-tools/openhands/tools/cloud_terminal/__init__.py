"""Cloud terminal tool package.

Provides a ``cloud_terminal`` tool that runs a single shell command through the
conversation's workspace and returns a structured result with captured stdout
and stderr. It works identically on a local workspace and on a remote or cloud
runtime (both implement ``BaseWorkspace.execute_command``), which is what lets
a DeepSeek-driven agent test an app in the cloud terminal.

Usage:
    from openhands.tools.cloud_terminal import CloudTerminalTool

    agent = Agent(
        llm=llm,
        tools=[Tool(name=CloudTerminalTool.name)],
    )
"""

from openhands.tools.cloud_terminal.definition import (
    CloudTerminalAction,
    CloudTerminalObservation,
    CloudTerminalTool,
)
from openhands.tools.cloud_terminal.impl import CloudTerminalExecutor


__all__ = [
    "CloudTerminalAction",
    "CloudTerminalObservation",
    "CloudTerminalTool",
    "CloudTerminalExecutor",
]