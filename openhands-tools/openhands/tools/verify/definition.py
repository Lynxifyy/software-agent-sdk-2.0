"""Verify tool: run a project's build/test command and report pass or fail.

Gives the agent a fast, structured "does this actually work?" check it can run
after editing code, instead of only reasoning about the change. It detects the
project type, runs the matching command (or one the agent supplies) in the
conversation's working directory, and reports the exit code plus captured
output.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
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


# Ordered most-specific first: a repo with several markers is classified by the
# first one found, so a Python project that also has a package.json is Python.
PROJECT_MARKERS: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    ("python", ("pyproject.toml", "setup.py", "setup.cfg", "requirements.txt")),
    ("node", ("package.json",)),
    ("rust", ("Cargo.toml",)),
    ("go", ("go.mod",)),
    ("java", ("pom.xml", "build.gradle", "build.gradle.kts")),
    ("ruby", ("Gemfile",)),
)

# The default command per project type. Deliberately conservative: the goal is a
# quick correctness signal, not a full CI run.
DEFAULT_COMMANDS: Final[dict[str, str]] = {
    "python": "python -m pytest -q",
    "node": "npm test --silent",
    "rust": "cargo test",
    "go": "go test ./...",
    "java": "mvn -q test",
    "ruby": "bundle exec rake test",
}


def detect_project_type(working_dir: Path) -> str:
    """Classify the project rooted at ``working_dir``, or "unknown"."""
    for name, markers in PROJECT_MARKERS:
        if any((working_dir / marker).exists() for marker in markers):
            return name
    return "unknown"


def resolve_command(project_type: str, override: str | None = None) -> str | None:
    """The command to run: the agent's override, else the type's default."""
    if override and override.strip():
        return override.strip()
    return DEFAULT_COMMANDS.get(project_type)


class VerifyAction(Action):
    """Run a project's build/test command to check the current code."""

    command: str | None = Field(
        default=None,
        description=(
            "Command to run. Omit to use the detected project's default "
            "(for example `npm test` for a Node project or `pytest` for Python)."
        ),
    )
    timeout_seconds: int | None = Field(
        default=None,
        description=(
            "Wall-clock limit for the command. Defaults to the tool's own limit."
        ),
    )

    @field_validator("timeout_seconds")
    @classmethod
    def _positive_timeout(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("timeout_seconds must be positive")
        return value

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("Verify: ", style="bold cyan")
        content.append(self.command or "auto-detected command")
        return content


class VerifyObservation(Observation):
    """Result of a verification run."""

    project_type: str = Field(default="unknown")
    command: str = Field(default="")
    exit_code: int | None = Field(default=None)
    passed: bool = Field(default=False)
    output: str = Field(default="")
    truncated: bool = Field(default=False)

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("Verification could not run", style="bold red")
        elif self.passed:
            content.append("Verification passed", style="bold green")
        else:
            content.append("Verification failed", style="bold red")
        if self.command:
            content.append(f"\n$ {self.command}")
        if self.output:
            content.append("\n")
            content.append(self.output)
        return content


_DESCRIPTION = (
    "Run the project's build or test command to check whether the code works, "
    "and get a structured pass/fail result.\n\n"
    "Use this after editing code to confirm the change before you report it "
    "done — a real build or test run is stronger evidence than reasoning about "
    "the diff. It runs in the conversation's working directory.\n\n"
    "By default the tool detects the project type (Python, Node, Rust, Go, "
    "Java, Ruby) and runs its standard test command. Pass `command` to run "
    "something specific (a single test file, a linter, a build script).\n\n"
    "The result reports the project type, the command, the exit code, whether "
    "it passed, and the captured output (truncated when very long). A non-zero "
    "exit is a normal failed check, not a tool error; a tool error means the "
    "command could not be run at all."
)


class VerifyTool(ToolDefinition[VerifyAction, VerifyObservation]):
    """Tool for running a project's verification command."""

    user_selectable: ClassVar[bool] = False

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,
        **params: Any,
    ) -> Sequence[Self]:
        if params:
            raise ValueError("VerifyTool does not accept parameters")

        from openhands.tools.verify.impl import VerifyExecutor

        working_dir = (
            conv_state.workspace.working_dir
            if conv_state is not None
            else str(Path.cwd())
        )

        return [
            cls(
                description=_DESCRIPTION,
                action_type=VerifyAction,
                observation_type=VerifyObservation,
                executor=VerifyExecutor(working_dir=working_dir),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=False,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            )
        ]


# Automatically register when this module is imported.
register_tool(VerifyTool.name, VerifyTool)
