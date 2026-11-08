"""Tests for the cloud_terminal tool: workspace execution with structured output."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from openhands.sdk.workspace.base import BaseWorkspace
from openhands.sdk.workspace.models import CommandResult
from openhands.tools.cloud_terminal.definition import (
    CloudTerminalAction,
    CloudTerminalObservation,
    CloudTerminalTool,
)
from openhands.tools.cloud_terminal.impl import CloudTerminalExecutor


@dataclass
class FakeWorkspace:
    """A workspace double that records commands; subclasses BaseWorkspace's shape."""

    working_dir: str = "/sandbox"

    def execute_command(self, command, cwd=None, timeout=30.0) -> CommandResult:
        raise NotImplementedError


class ScriptedWorkspace(FakeWorkspace):
    """Returns scripted results keyed by command text."""

    def __init__(self, results: dict[str, CommandResult]) -> None:
        self.results = results

    def execute_command(self, command, cwd=None, timeout=30.0) -> CommandResult:
        recorded = dict(self.results)
        # A sentinel fallback makes "success" calls easy to write.
        for key, result in recorded.items():
            if key in command:
                return result
        return CommandResult(
            command=command, exit_code=0, stdout="", stderr="", timeout_occurred=False
        )


def _observation(command: str = "pytest -q", result: CommandResult | None = None) -> CloudTerminalObservation:
    if result is None:
        result = CommandResult(
            command=command,
            exit_code=0,
            stdout="1 passed",
            stderr="",
            timeout_occurred=False,
        )
    workspace = ScriptedWorkspace({command: result})
    executor = CloudTerminalExecutor(workspace=workspace)  # type: ignore[arg-type]
    return executor(CloudTerminalAction(command=command))


class TestActionSchema:
    def test_requires_a_command(self) -> None:
        with pytest.raises(ValueError):
            CloudTerminalAction(command="   ")

    def test_rejects_a_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError):
            CloudTerminalAction(command="ls", timeout_seconds=0)


class TestToolDefinition:
    def test_registers_under_the_expected_name(self) -> None:
        assert CloudTerminalTool.name == "cloud_terminal"

    def test_create_requires_a_conversation(self) -> None:
        with pytest.raises(ValueError):
            CloudTerminalTool.create()


class TestExecutor:
    def test_reports_success_with_stdout_and_exit_code(self) -> None:
        observation = _observation()
        assert observation.is_error is False
        assert observation.exit_code == 0
        assert observation.stdout == "1 passed"
        assert observation.stderr == ""

    def test_reports_failure_with_stderr(self) -> None:
        result = CommandResult(
            command="pytest -q",
            exit_code=1,
            stdout="1 failed",
            stderr="tests/test_x.py:3: AssertionError",
            timeout_occurred=False,
        )
        observation = _observation(result=result)
        assert observation.is_error is False
        assert observation.exit_code == 1
        assert observation.stderr.startswith("tests/test_x.py")

    def test_marks_a_timeout(self) -> None:
        result = CommandResult(
            command="sleep 999",
            exit_code=124,
            stdout="",
            stderr="",
            timeout_occurred=True,
        )
        observation = _observation(command="sleep 999", result=result)
        assert observation.timeout_occurred is True

    def test_truncates_large_output(self) -> None:
        blob = "x" * 20_000
        result = CommandResult(
            command="cat big.txt",
            exit_code=0,
            stdout=blob,
            stderr="",
            timeout_occurred=False,
        )
        observation = _observation(command="cat big.txt", result=result)
        assert len(observation.stdout) < len(blob)
        assert observation.truncated is True

    def test_reports_a_workspace_failure_as_an_error(self) -> None:
        class ExplodingWorkspace(FakeWorkspace):
            def execute_command(self, command, cwd=None, timeout=30.0) -> CommandResult:
                raise OSError("sandbox unreachable")

        executor = CloudTerminalExecutor(workspace=ExplodingWorkspace())  # type: ignore[arg-type]
        observation = executor(CloudTerminalAction(command="ls"))
        assert observation.is_error is True
        assert "sandbox unreachable" in observation.text
