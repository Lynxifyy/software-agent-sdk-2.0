"""Cloud terminal tool: run one command through the conversation's workspace.

Distinct from the persistent ``terminal`` session, this is a one-shot structured
runner: it executes a single command in the conversation's working directory via
``BaseWorkspace.execute_command`` and returns a structured result (exit code,
stdout, stderr, whether it timed out). Both local and remote/cloud workspaces
implement that method, so the same tool drives a locally-persisted sandbox or a
cloud runtime — letting a model like DeepSeek test an app in the cloud terminal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final, Self

from pydantic import Field, field_validator
from rich.text import Text

from openhands.sdk.tool.registry import register_tool
from openhands.sdk.tool.tool import (
    Action,
    Observation,
    ToolAnnotations,
    ToolDefinition,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation.state import ConversationState

DEFAULT_TIMEOUT_SECONDS: Final[int] = 120
MAX_OUTPUT_CHARS: Final[int] = 16_000


class CloudTerminalAction(Action):
    """Run a single shell command in the conversation's working directory."""

    command: str = Field(
        description=(
            "The shell command to run. Runs in the conversation's working "
            "directory, so relative paths and project scripts resolve there."
        )
    )
    timeout_seconds: int | None = Field(
        default=None,
        description=(
            "Wall-clock limit for the command. Defaults to the tool's own limit."
        ),
    )

    @field_validator("command")
    @classmethod
    def _non_empty_command(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("command must not be empty")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def _positive_timeout(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("timeout_seconds must be positive")
        return value

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Cloud terminal: ", style="bold blue")
        content.append(self.command)
        return content


class CloudTerminalObservation(Observation):
    """Result of a cloud-terminal command run."""

    command: str = Field(default="")
    exit_code: int | None = Field(default=None)
    stdout: str = Field(default="")
    stderr: str = Field(default="")
    timeout_occurred: bool = Field(default=False)
    truncated: bool = Field(default=False)

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Cloud command could not run", style="bold red")
        elif self.exit_code == 0:
            content.append("Command succeeded", style="bold green")
        else:
            content.append(
                f"Command failed (exit {self.exit_code})", style="bold red"
            )
        content.append(f"\n$ {self.command}")
        if self.stdout:
            content.append("\n")
            content.append(self.stdout)
        if self.stderr:
            content.append("\n[stderr]\n")
            content.append(self.stderr)
        if self.truncated:
            content.append("\n… [output truncated]")
        return content


_DESCRIPTION = (
    "Run one shell command in the cloud terminal and get a structured result "
    "(exit code, stdout, stderr).\n\n"
    "Use this to build, test, or exercise the app directly in its sandbox — "
    "the command runs in the conversation's working directory in the cloud "
    "runtime, so it sees the same filesystem the agent edits.\n\n"
    "Prefer this over the persistent `terminal` session for quick, one-shot "
    "checks where you want a clean pass/fail answer (running a test file, a "
    "build step, a script). It returns the captured output, truncated when "
    "very long, plus the exit code and whether the command timed out."
)


class CloudTerminalTool(ToolDefinition[CloudTerminalAction, CloudTerminalObservation]):
    """Tool for running a single command through the conversation workspace."""

    user_selectable: ClassVar[bool] = False

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,
        **params: Any,
    ) -> Sequence[Self]:
        if params:
            raise ValueError("CloudTerminalTool does not accept parameters")

        from openhands.tools.cloud_terminal.impl import CloudTerminalExecutor

        if conv_state is None:
            raise ValueError(
                "CloudTerminalTool requires a conversation to run commands in"
            )

        return [
            cls(
                description=_DESCRIPTION,
                action_type=CloudTerminalAction,
                observation_type=CloudTerminalObservation,
                executor=CloudTerminalExecutor(workspace=conv_state.workspace),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            )
        ]


# Automatically register when this module is imported.
register_tool(CloudTerminalTool.name, CloudTerminalTool)
