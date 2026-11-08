"""File builder executor: write text/PNG/ZIP/DOCX artifacts to disk."""

from __future__ import annotations

import base64
import io
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Final

from openhands.sdk.logger import get_logger
from openhands.sdk.tool import ToolExecutor
from openhands.tools.file_builder.definition import (
    FileBuilderAction,
    FileBuilderObservation,
    ZipEntry,
)


if TYPE_CHECKING:
    from openhands.sdk.conversation import LocalConversation


logger = get_logger(__name__)

_ESCAPES: Final[dict[str, str]] = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
}


def _xml_escape(value: str) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in value)


class FileBuilderExecutor(ToolExecutor[FileBuilderAction, FileBuilderObservation]):
    """Create artifacts in a working directory."""

    def __init__(self, working_dir: str) -> None:
        self.working_dir: Path = Path(working_dir).resolve()

    def _resolve(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            resolved = candidate.resolve()
        else:
            resolved = (self.working_dir / candidate).resolve()
        # Keep writes inside the sandbox working directory.
        if not resolved.is_relative_to(self.working_dir):
            raise ValueError(f"path resolves outside the working directory: {path}")
        return resolved

    def _write_text(self, path: Path, content: str) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path.stat().st_size

    def _write_png(self, path: Path, content: str) -> int:
        raw = base64.b64decode(content, validate=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return path.stat().st_size

    def _write_zip(self, path: Path, entries: list[ZipEntry]) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for entry in entries:
                archive.writestr(entry.name, entry.content)
        return path.stat().st_size

    def _write_docx(self, path: Path, paragraphs: list[str]) -> int:
        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/'
            'wordprocessingml/2006/main">'
            "<w:body>"
        )
        for paragraph in paragraphs:
            text = _xml_escape(paragraph)
            document_xml += (
                "<w:p><w:r><w:t>" + text + "</w:t></w:r></w:p>"
            )
        document_xml += "</w:body></w:document>"

        path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(
                "[Content_Types].xml",
                (
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/'
                    'content-types">'
                    '<Default Extension="rels" ContentType="application/vnd.'
                    'openxmlformats-package.relationships+xml"/>'
                    '<Default Extension="xml" ContentType="application/xml"/>'
                    '<Override PartName="/word/document.xml" ContentType="application/'
                    'vnd.openxmlformats-officedocument.wordprocessingml.document.'
                    'main+xml"/>'
                    "</Types>"
                ),
            )
            archive.writestr(
                "_rels/.rels",
                (
                    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                    '<Relationships xmlns="http://schemas.openxmlformats.org/'
                    'package/2006/relationships">'
                    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/'
                    'officeDocument/2006/relationships/officeDocument" Target='
                    '"word/document.xml"/>'
                    "</Relationships>"
                ),
            )
            archive.writestr("word/document.xml", document_xml)
        return path.stat().st_size

    def __call__(
        self,
        action: FileBuilderAction,
        conversation: LocalConversation | None = None,  # noqa: ARG002
    ) -> FileBuilderObservation:
        try:
            path = self._resolve(action.path)
            if action.build_kind == "write_text":
                size = self._write_text(path, action.content)
            elif action.build_kind == "write_png":
                size = self._write_png(path, action.content)
            elif action.build_kind == "write_zip":
                size = self._write_zip(path, action.entries)
            else:
                size = self._write_docx(path, action.paragraphs)
        except Exception as error:  # noqa: BLE001 - surface any build failure
            logger.warning("file_builder failed to create %r: %s", action.path, error)
            return FileBuilderObservation.from_text(
                text=f"Could not create '{action.path}': {error}",
                is_error=True,
                build_kind=action.build_kind,
                path=action.path,
                bytes_written=0,
            )

        return FileBuilderObservation.from_text(
            text=f"Wrote {size} bytes to {path}",
            is_error=False,
            build_kind=action.build_kind,
            path=str(path),
            bytes_written=size,
        )