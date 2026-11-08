"""File builder tool package.

Provides a ``file_builder`` tool that creates real artifacts — text/markdown
files, ZIP archives, PNG images (from base64 bytes) and minimal DOCX documents
— in the conversation's working directory. This is the artifact-making side of
the Oli coding agent.

Usage:
    from openhands.tools.file_builder import FileBuilderTool

    agent = Agent(
        llm=llm,
        tools=[Tool(name=FileBuilderTool.name)],
    )
"""

from openhands.tools.file_builder.definition import (
    FileBuilderAction,
    FileBuilderObservation,
    FileBuilderTool,
)
from openhands.tools.file_builder.impl import FileBuilderExecutor


__all__ = [
    "FileBuilderAction",
    "FileBuilderObservation",
    "FileBuilderTool",
    "FileBuilderExecutor",
]