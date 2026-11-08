"""Cloud terminal executor: run a command through the workspace."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.sdk.utils import maybe_truncate
from openhands.sdk.workspace.base import BaseWorkspace
from openhands.tools.cloud_terminal.definition import (
    DEFAULT_TIMEOUT_SECONDS,
    MAX_OUTPUT_CHARS,
    CloudTerminalAction,
    CloudTerminalObservation,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


logger = get_logger(__name__)

_TRUNCATE_NOTICE: Final[str] = "\n… [output truncated] …\n"


class CloudTerminalExecutor(ToolExecutor[CloudTerminalAction, CloudTerminalObservation]):
    """Execute a command via the conversation workspace and structure the result."""

    def __init__(
        self,
        workspace: BaseWorkspace,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = MAX_OUTPUT_CHARS,
    ) -> None:
        self.workspace = workspace
        self.timeout_seconds = timeout_seconds
        self.max_output_chars = max_output_chars

    def _truncate(self, text: str) -> tuple[str, bool]:
        if len(text) <= self.max_output_chars:
            return text, False
        truncated = maybe_truncate(
            text,
            truncate_after=self.max_output_chars,
            truncate_notice=_TRUNCATE_NOTICE,
        )
        return truncated, True

    def __call__(
        self,
        action: CloudTerminalAction,
        conversation: LocalConversation | None = None,  # noqa: ARG002
    ) -> CloudTerminalObservation:
        command = action.command.strip()
        timeout = action.timeout_seconds or self.timeout_seconds

        try:
            result = self.workspace.execute_command(
                command,
                timeout=timeout,
            )
        except Exception as error:  # noqa: BLE001 - surface any workspace failure
            logger.warning("cloud_terminal failed to run %r: %s", command, error)
            return CloudTerminalObservation.from_text(
                text=f"Could not run '{command}': {error}",
                is_error=True,
                command=command,
                exit_code=None,
                stdout="",
                stderr="",
                timeout_occurred=False,
                truncated=False,
            )

        stdout, stdout_truncated = self._truncate(result.stdout or "")
        stderr, stderr_truncated = self._truncate(result.stderr or "")
        truncated = stdout_truncated or stderr_truncated

        summary = "succeeded" if result.exit_code == 0 else f"failed (exit {result.exit_code})"
        return CloudTerminalObservation.from_text(
            text=f"Command {summary}: {command}",
            is_error=False,
            command=command,
            exit_code=result.exit_code,
            stdout=stdout,
            stderr=stderr,
            timeout_occurred=result.timeout_occurred,
            truncated=truncated,
        )