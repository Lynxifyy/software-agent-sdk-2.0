"""Tests for the file_builder tool: text, PNG, ZIP and DOCX artifact creation."""

from __future__ import annotations

import base64
import zipfile
from pathlib import Path

import pytest

from openhands.tools.file_builder.definition import (
    FileBuilderAction,
    FileBuilderObservation,
    FileBuilderTool,
    ZipEntry,
)
from openhands.tools.file_builder.impl import FileBuilderExecutor


def _executor(tmp_path: Path) -> FileBuilderExecutor:
    return FileBuilderExecutor(working_dir=str(tmp_path))


class TestActionSchema:
    def test_requires_content_for_text(self) -> None:
        with pytest.raises(ValueError):
            FileBuilderAction(build_kind="write_text", path="a.md")

    def test_requires_a_zip_entry(self) -> None:
        with pytest.raises(ValueError):
            FileBuilderAction(build_kind="write_zip", path="a.zip")

    def test_rejects_an_unsafe_zip_entry(self) -> None:
        with pytest.raises(ValueError):
            FileBuilderAction(
                build_kind="write_zip",
                path="a.zip",
                entries=[ZipEntry(name="../escape.txt", content="x")],
            )


class TestToolDefinition:
    def test_registers_under_the_expected_name(self) -> None:
        assert FileBuilderTool.name == "file_builder"

    def test_create_requires_a_conversation(self) -> None:
        with pytest.raises(ValueError):
            FileBuilderTool.create()


class TestExecutor:
    def test_writes_a_text_file(self, tmp_path: Path) -> None:
        observation = _executor(tmp_path)(
            FileBuilderAction(
                build_kind="write_text",
                path="report.md",
                content="# Report\n\nDone.",
            )
        )
        assert observation.is_error is False
        assert observation.bytes_written > 0
        assert (tmp_path / "report.md").read_text(encoding="utf-8") == (
            "# Report\n\nDone."
        )

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        _executor(tmp_path)(
            FileBuilderAction(build_kind="write_text", path="docs/notes.txt", content="x")
        )
        assert (tmp_path / "docs" / "notes.txt").read_text() == "x"

    def test_writes_a_png_from_base64(self, tmp_path: Path) -> None:
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
        observation = _executor(tmp_path)(
            FileBuilderAction(
                build_kind="write_png",
                path="img.png",
                content=base64.b64encode(png_bytes).decode("ascii"),
            )
        )
        assert observation.is_error is False
        assert (tmp_path / "img.png").read_bytes() == png_bytes

    def test_creates_a_zip_archive(self, tmp_path: Path) -> None:
        observation = _executor(tmp_path)(
            FileBuilderAction(
                build_kind="write_zip",
                path="bundle.zip",
                entries=[
                    ZipEntry(name="a.txt", content="aaa"),
                    ZipEntry(name="nested/b.txt", content="bbb"),
                ],
            )
        )
        assert observation.is_error is False
        with zipfile.ZipFile(tmp_path / "bundle.zip") as archive:
            assert archive.read("a.txt") == b"aaa"
            assert archive.read("nested/b.txt") == b"bbb"

    def test_creates_a_docx(self, tmp_path: Path) -> None:
        observation = _executor(tmp_path)(
            FileBuilderAction(
                build_kind="write_docx",
                path="out.docx",
                paragraphs=["Hello <world>", "Second & final"],
            )
        )
        assert observation.is_error is False
        with zipfile.ZipFile(tmp_path / "out.docx") as archive:
            document_xml = archive.read("word/document.xml").decode("utf-8")
            assert "Hello &lt;world&gt;" in document_xml
            assert "Second &amp; final" in document_xml

    def test_rejects_a_path_outside_the_working_directory(self, tmp_path: Path) -> None:
        observation = _executor(tmp_path)(
            FileBuilderAction(build_kind="write_text", path="/etc/escape", content="x")
        )
        assert observation.is_error is True
        assert "outside the working directory" in observation.text
