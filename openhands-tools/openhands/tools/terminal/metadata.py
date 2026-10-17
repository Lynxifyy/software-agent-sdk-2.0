"""Metadata for bash command execution."""

import json
import re
import traceback

from pydantic import BaseModel, Field

from openhands.sdk.logger import get_logger
from openhands.tools.terminal.constants import (
    CMD_OUTPUT_METADATA_PS1_REGEX,
    CMD_OUTPUT_PS1_BEGIN,
    CMD_OUTPUT_PS1_END,
)


logger = get_logger(__name__)


class CmdOutputMetadata(BaseModel):
    """Additional metadata captured from PS1"""

    exit_code: int = Field(
        default=-1, description="The exit code of the last executed command."
    )
    pid: int = Field(
        default=-1, description="The process ID of the last executed command."
    )
    username: str | None = Field(
        default=None, description="The username of the current user."
    )
    hostname: str | None = Field(
        default=None, description="The hostname of the machine."
    )
    working_dir: str | None = Field(
        default=None, description="The current working directory."
    )
    py_interpreter_path: str | None = Field(
        default=None, description="The path to the current Python interpreter, if any."
    )
    prefix: str = Field(default="", description="Prefix to add to command output")
    suffix: str = Field(default="", description="Suffix to add to command output")

    @classmethod
    def to_ps1_prompt(cls, *, escape: bool = True) -> str:
        """Convert the required metadata into a PS1 prompt.

        Args:
            escape: Escape the JSON double quotes. Required when the prompt is
                assigned through a double-quoted ``export PS1="..."`` in
                ``PROMPT_COMMAND`` (tmux), where bash strips the quotes. A shell
                that keeps ``PS1`` literal — the PTY subprocess backend, and any
                non-bash shell — must pass ``escape=False``, or the backslashes
                are printed and the metadata stops being valid JSON.
        """
        prompt = CMD_OUTPUT_PS1_BEGIN
        json_str = json.dumps(
            {
                "pid": "$!",
                "exit_code": "$?",
                # ``\u``/``\h`` are bash-only PS1 escapes; on a POSIX sh they stay
                # literal and ``\u`` is not even valid JSON, so the metadata stops
                # parsing. Command/parameter expansion works in both shells.
                "username": "${USER}",
                # No single quotes: tmux assigns this prompt inside a
                # single-quoted PROMPT_COMMAND, where a quote would end the
                # assignment early.
                "hostname": "$(hostname 2>/dev/null)",
                "working_dir": r"$(pwd)",
                "py_interpreter_path": r'$(command -v python || echo "")',
            },
            indent=2,
        )
        # Make sure we escape double quotes in the JSON string
        # So that PS1 will keep them as part of the output
        prompt += json_str.replace('"', r"\"") if escape else json_str
        prompt += CMD_OUTPUT_PS1_END + "\n"  # Ensure there's a newline at the end
        return prompt

    @classmethod
    def matches_ps1_metadata(cls, string: str) -> list[re.Match[str]]:
        """Find all valid PS1 metadata blocks in the string."""
        matches: list[re.Match[str]] = []
        for match in CMD_OUTPUT_METADATA_PS1_REGEX.finditer(string):
            content = match.group(1).strip()
            try:
                json.loads(content)
                matches.append(match)
            except json.JSONDecodeError:
                logger.debug(
                    f"Failed to parse PS1 metadata - Skipping: [{content[:200]}"
                    f"{'...' if len(content) > 200 else ''}]" + traceback.format_exc()
                )
        return matches

    @classmethod
    def from_ps1_match(cls, match: re.Match[str]) -> "CmdOutputMetadata":
        """Extract the required metadata from a PS1 prompt."""
        metadata = json.loads(match.group(1))
        # Create a copy of metadata to avoid modifying the original
        processed = metadata.copy()
        # Convert numeric fields
        if "pid" in metadata:
            try:
                processed["pid"] = int(float(str(metadata["pid"])))
            except (ValueError, TypeError):
                processed["pid"] = -1
        if "exit_code" in metadata:
            try:
                processed["exit_code"] = int(float(str(metadata["exit_code"])))
            except (ValueError, TypeError):
                logger.debug(
                    f"Failed to parse exit code: {metadata['exit_code']}. "
                    f"Setting to -1."
                )
                processed["exit_code"] = -1
        return cls(**processed)
