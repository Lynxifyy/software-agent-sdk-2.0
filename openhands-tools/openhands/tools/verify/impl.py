"""Verify tool executor: run the project command and capture the result."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Final

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.tools.verify.definition import (
    VerifyAction,
    VerifyObservation,
    detect_project_type,
    resolve_command,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


logger = get_logger(__name__)

DEFAULT_TIMEOUT_SECONDS: Final[int] = 600
DEFAULT_MAX_OUTPUT_CHARS: Final[int] = 16_000


class VerifyExecutor(ToolExecutor[VerifyAction, VerifyObservation]):
    """Detect the project type and run its verification command."""

    def __init__(
        self,
        working_dir: str,
        commands: dict[str, str] | None = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self.working_dir: Path = Path(working_dir).resolve()
        self.commands: dict[str, str] | None = commands
        self.timeout_seconds: int = timeout_seconds
        self.max_output_chars: int = max_output_chars

    def _resolve_command(self, project_type: str, override: str | None) -> str | None:
        if override and override.strip():
            return override.strip()
        if self.commands is not None:
            return self.commands.get(project_type)
        return resolve_command(project_type, None)

    def __call__(
        self,
        action: VerifyAction,
        conversation: LocalConversation | None = None,  # noqa: ARG002
    ) -> VerifyObservation:
        project_type = detect_project_type(self.working_dir)
        command = self._resolve_command(project_type, action.command)

        if not command:
            return VerifyObservation.from_text(
                text=(
                    "Could not determine a verification command: the project "
                    f"type is '{project_type}' and no command was given. Pass "
                    "`command` with the build or test command to run."
                ),
                is_error=True,
                project_type=project_type,
                command="",
                exit_code=None,
                passed=False,
                output="",
                truncated=False,
            )

        timeout = action.timeout_seconds or self.timeout_seconds
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.working_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            partial = _decode(error.stdout) + _decode(error.stderr)
            note = f"Command timed out after {timeout}s: {command}\n"
            output, truncated = self._truncate(note + partial)
            return VerifyObservation.from_text(
                text=f"Command timed out after {timeout}s: {command}",
                is_error=True,
                project_type=project_type,
                command=command,
                exit_code=None,
                passed=False,
                output=output,
                truncated=truncated,
            )
        except OSError as error:
            return VerifyObservation.from_text(
                text=f"Could not run '{command}': {error}",
                is_error=True,
                project_type=project_type,
                command=command,
                exit_code=None,
                passed=False,
                output="",
                truncated=False,
            )

        combined = (completed.stdout or "") + (completed.stderr or "")
        output, truncated = self._truncate(combined)
        passed = completed.returncode == 0
        summary = "passed" if passed else f"failed (exit {completed.returncode})"
        return VerifyObservation.from_text(
            text=f"Verification {summary}: {command}",
            is_error=False,
            project_type=project_type,
            command=command,
            exit_code=completed.returncode,
            passed=passed,
            output=output,
            truncated=truncated,
        )

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        marker = "\n... [output truncated] ...\n"
        # Keep the tail: the end of a test run holds the failure summary. The
        # marker is part of the budget so the result never exceeds the limit.
        budget = max(self.max_output_chars - len(marker), 0)
        head = budget // 4
        tail = budget - head
        return text[:head] + marker + text[len(text) - tail :], True


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value
