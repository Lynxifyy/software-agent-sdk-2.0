"""File builder tool: create text, markdown, PNG, ZIP and DOCX artifacts.

The action carries a ``build_kind`` so schemas stay a closed union instead of several
almost-identical tools. The executor resolves the path against the conversation
working directory (absolute paths stay absolute; relative paths resolve under
``working_dir``), writes the artifact, and returns the created file's size.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar, Final, Literal, Self

from pydantic import Field, field_validator, model_validator
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


MAX_PAYLOAD_CHARS: Final[int] = 2_000_000
MAX_ZIP_ENTRIES: Final[int] = 2_000


class ZipEntry(Action):  # noqa: SLOT001 - reused as a nested model
    """One file inside a ZIP archive produced by ``file_builder``."""

    name: str = Field(description="Path of the entry inside the archive.")
    content: str = Field(default="", description="UTF-8 text content of the entry.")


class FileBuilderAction(Action):
    """Create a text, markdown, PNG, ZIP or DOCX artifact."""

    build_kind: Literal["write_text", "write_png", "write_zip", "write_docx"] = Field(
        description=(
            "What to create: write_text for .txt/.md/.json/anything textual, "
            "write_png for an image (content is base64 PNG bytes), write_zip "
            "for an archive of text entries, write_docx for a Word document "
            "built from paragraphs."
        )
    )
    path: str = Field(
        description=(
            "Destination path. Absolute paths are used as-is; relative paths "
            "resolve inside the conversation's working directory. The parent "
            "directory is created if missing."
        )
    )
    content: str = Field(
        default="",
        description=(
            "For write_text: the file content. For write_png: base64-encoded "
            "PNG bytes. Unused for write_zip/write_docx."
        ),
    )
    entries: list[ZipEntry] = Field(
        default_factory=list,
        description=(
            "write_zip only: the files to pack. Each entry is a name and its "
            "UTF-8 text content."
        ),
    )
    paragraphs: list[str] = Field(
        default_factory=list,
        description="write_docx only: the document's paragraphs.",
    )

    @field_validator("content")
    @classmethod
    def _bounded_content(cls, value: str) -> str:
        if len(value) > MAX_PAYLOAD_CHARS:
            raise ValueError("content is too large")
        return value

    @field_validator("entries")
    @classmethod
    def _bounded_entries(cls, value: list[ZipEntry]) -> list[ZipEntry]:
        if len(value) > MAX_ZIP_ENTRIES:
            raise ValueError("too many zip entries")
        for entry in value:
            if not entry.name.strip():
                raise ValueError("zip entry names must not be empty")
            if "\x00" in entry.name or ".." in entry.name.split("/"):
                raise ValueError(f"unsafe zip entry name: {entry.name!r}")
        return value

    @field_validator("paragraphs")
    @classmethod
    def _bounded_paragraphs(cls, value: list[str]) -> list[str]:
        if sum(len(p) for p in value) > MAX_PAYLOAD_CHARS:
            raise ValueError("paragraphs are too large")
        return value

    @model_validator(mode="after")
    def _require_content_for_write_text(self) -> "FileBuilderAction":
        if self.build_kind in ("write_text", "write_png") and not self.content:
            raise ValueError(f"{self.build_kind} requires non-empty content")
        if self.build_kind == "write_zip" and not self.entries:
            raise ValueError("write_zip requires at least one entry")
        return self

    @property
    def visualize(self) -> Text:
        content = Text()
        content.append("File builder: ", style="bold blue")
        content.append(self.build_kind)
        content.append(" ")
        content.append(self.path)
        return content


class FileBuilderObservation(Observation):
    """Result of a file-build operation."""

    build_kind: Literal[
        "write_text", "write_png", "write_zip", "write_docx"
    ] = Field(default="write_text")
    path: str = Field(default="")
    bytes_written: int = Field(default=0)

    @property
    def visualize(self) -> Text:
        content = Text()
        if self.is_error:
            content.append("File not created", style="bold red")
        else:
            content.append("File created", style="bold green")
        content.append(f" ({self.bytes_written} bytes): {self.path}")
        return content


_DESCRIPTION = (
    "Create a real file artifact in the conversation's working directory: a "
    "text or markdown document, a PNG image, a ZIP archive, or a Word (DOCX) "
    "document.\n\n"
    "Use this when the task asks for a deliverable file (a README, a report, "
    "an image, a bundle of files, a document). It writes the file to disk and "
    "reports the bytes written so the caller knows it exists.\n\n"
    "Kinds:\n"
    "  - write_text: `path` + `content` → a .txt/.md/.json/… text file.\n"
    "  - write_png: `path` + `content` (base64-encoded PNG bytes) → an image.\n"
    "  - write_zip: `path` + `entries` (name + UTF-8 text content each) → a ZIP.\n"
    "  - write_docx: `path` + `paragraphs` → a minimal Word document.\n\n"
    "Relative paths resolve inside the conversation's working directory; "
    "parent directories are created automatically."
)


class FileBuilderTool(ToolDefinition[FileBuilderAction, FileBuilderObservation]):
    """Tool for creating file artifacts (text, PNG, ZIP, DOCX)."""

    user_selectable: ClassVar[bool] = False

    @classmethod
    def create(
        cls,
        conv_state: ConversationState | None = None,
        **params: Any,
    ) -> Sequence[Self]:
        if params:
            raise ValueError("FileBuilderTool does not accept parameters")

        from openhands.tools.file_builder.impl import FileBuilderExecutor

        if conv_state is None:
            raise ValueError(
                "FileBuilderTool requires a conversation to write files in"
            )

        return [
            cls(
                description=_DESCRIPTION,
                action_type=FileBuilderAction,
                observation_type=FileBuilderObservation,
                executor=FileBuilderExecutor(working_dir=conv_state.workspace.working_dir),
                annotations=ToolAnnotations(
                    readOnlyHint=False,
                    destructiveHint=True,
                    idempotentHint=False,
                    openWorldHint=False,
                ),
            )
        ]


# Automatically register when this module is imported.
register_tool(FileBuilderTool.name, FileBuilderTool)
