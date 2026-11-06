"""Tests for the verify tool: project detection and command execution."""

from __future__ import annotations

from pathlib import Path

import pytest

from openhands.tools.verify.definition import (
    VerifyAction,
    VerifyObservation,
    VerifyTool,
    detect_project_type,
    resolve_command,
)
from openhands.tools.verify.impl import VerifyExecutor


class TestDetection:
    def test_detects_node_from_package_json(self, tmp_path: Path) -> None:
        (tmp_path / "package.json").write_text("{}")
        assert detect_project_type(tmp_path) == "node"

    def test_detects_python_from_pyproject(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        assert detect_project_type(tmp_path) == "python"

    def test_detects_rust_from_cargo_toml(self, tmp_path: Path) -> None:
        (tmp_path / "Cargo.toml").write_text("[package]\n")
        assert detect_project_type(tmp_path) == "rust"

    def test_detects_go_from_go_mod(self, tmp_path: Path) -> None:
        (tmp_path / "go.mod").write_text("module x\n")
        assert detect_project_type(tmp_path) == "go"

    def test_returns_unknown_for_an_empty_dir(self, tmp_path: Path) -> None:
        assert detect_project_type(tmp_path) == "unknown"

    def test_prefers_a_more_specific_marker(self, tmp_path: Path) -> None:
        # A repo with both a pyproject and a package.json: the pyproject at the
        # root wins because it is listed first.
        (tmp_path / "pyproject.toml").write_text("[project]\n")
        (tmp_path / "package.json").write_text("{}")
        assert detect_project_type(tmp_path) == "python"


class TestResolveCommand:
    def test_uses_the_override_when_given(self) -> None:
        assert resolve_command("python", "pytest -q") == "pytest -q"

    def test_maps_known_types_to_a_command(self) -> None:
        assert resolve_command("node") is not None
        assert resolve_command("python") is not None

    def test_returns_none_for_unknown_without_an_override(self) -> None:
        assert resolve_command("unknown") is None


class TestActionSchema:
    def test_defaults(self) -> None:
        action = VerifyAction()
        assert action.command is None
        assert action.timeout_seconds is None

    def test_rejects_a_non_positive_timeout(self) -> None:
        with pytest.raises(ValueError):
            VerifyAction(timeout_seconds=0)


class TestToolDefinition:
    def test_registers_under_the_expected_name(self) -> None:
        assert VerifyTool.name == "verify"

    def test_create_returns_a_single_tool(self) -> None:
        tools = VerifyTool.create()
        assert len(tools) == 1
        assert tools[0].name == "verify"

    def test_is_a_mutating_network_tool(self) -> None:
        tool = VerifyTool.create()[0]
        assert tool.annotations is not None
        assert tool.annotations.readOnlyHint is False


class TestExecutor:
    def _run(self, tmp_path: Path, **kwargs) -> VerifyObservation:
        executor = VerifyExecutor(working_dir=str(tmp_path))
        return executor(VerifyAction(**kwargs))

    def test_reports_pass_for_a_successful_command(self, tmp_path: Path) -> None:
        observation = self._run(tmp_path, command="python3 -c \"print('ok')\"")
        assert observation.is_error is False
        assert observation.passed is True
        assert observation.exit_code == 0
        assert "ok" in observation.output

    def test_reports_failure_for_a_nonzero_exit(self, tmp_path: Path) -> None:
        observation = self._run(
            tmp_path, command='python3 -c "import sys; sys.exit(3)"'
        )
        assert observation.is_error is False
        assert observation.passed is False
        assert observation.exit_code == 3

    def test_refuses_an_unknown_project_without_a_command(self, tmp_path: Path) -> None:
        observation = self._run(tmp_path)
        assert observation.is_error is True
        assert observation.project_type == "unknown"

    def test_detects_and_runs_the_project_command(self, tmp_path: Path) -> None:
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        executor = VerifyExecutor(
            working_dir=str(tmp_path),
            commands={"python": "python3 -c \"print('ran')\""},
        )
        observation = executor(VerifyAction())
        assert observation.project_type == "python"
        assert observation.passed is True
        assert "ran" in observation.output

    def test_truncates_large_output(self, tmp_path: Path) -> None:
        executor = VerifyExecutor(working_dir=str(tmp_path), max_output_chars=100)
        observation = executor(VerifyAction(command="python3 -c \"print('x' * 5000)\""))
        assert len(observation.output) <= 100
        assert observation.truncated is True

    def test_times_out_a_hanging_command(self, tmp_path: Path) -> None:
        executor = VerifyExecutor(working_dir=str(tmp_path))
        observation = executor(
            VerifyAction(
                command='python3 -c "import time; time.sleep(30)"',
                timeout_seconds=1,
            ),
        )
        assert observation.is_error is True
        assert "timed out" in observation.output.lower()
