"""Braind Core - document text extraction.

Turns an uploaded file into plain text for the ingest pipeline. Formats that
are ZIP+XML containers (.docx, .pptx, .odt, .epub) are read with the standard
library; PDF and XLSX use ``pypdf`` and ``openpyxl``.

Extraction is deliberately shallow: it recovers readable prose, not layout.
Anything it cannot parse raises ``UnsupportedDocument`` so the caller can turn
that into a typed API error rather than ingesting binary noise.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from bs4 import BeautifulSoup

# Plain-text extensions ingested as-is.
TEXT_EXTENSIONS = frozenset(
    {
        ".md",
        ".txt",
        ".json",
        ".csv",
        ".tsv",
        ".log",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".htm",
        ".css",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".sql",
        ".sh",
        ".rst",
    }
)

_OOXML_NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
}
_ODF_TEXT_NS = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"

MIME_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.oasis.opendocument.text": ".odt",
    "application/epub+zip": ".epub",
}


class UnsupportedDocument(ValueError):
    """The file format is not supported, or the file could not be read."""


def _is_probable_binary(sample: bytes) -> bool:
    """Return whether a byte sample is probably not a text file."""
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    control = sum(1 for byte in sample[:512] if byte < 9 or 14 <= byte < 32)
    return control / max(len(sample[:512]), 1) > 0.12


def _zip_archive(data: bytes) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(data), "r")
    except zipfile.BadZipFile as exc:
        raise UnsupportedDocument("The file is not a readable archive.") from exc


def _extract_pdf(data: bytes) -> str:
    from pypdf import PdfReader
    from pypdf.errors import PyPdfError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            # An empty user password is common for "protected" but readable PDFs.
            try:
                reader.decrypt("")
            except Exception as exc:
                raise UnsupportedDocument("The PDF is password protected.") from exc
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
    except UnsupportedDocument:
        raise
    except (PyPdfError, ValueError, OSError) as exc:
        raise UnsupportedDocument("The PDF could not be read.") from exc
    return "\n\n".join(page for page in pages if page)


def _extract_docx(data: bytes) -> str:
    with _zip_archive(data) as archive:
        try:
            xml_data = archive.read("word/document.xml")
        except KeyError as exc:
            raise UnsupportedDocument("The .docx file has no document body.") from exc
    root = _parse_xml(xml_data)
    paragraphs = []
    for node in root.iter(f"{{{_OOXML_NS['w']}}}p"):
        paragraph = "".join(node.itertext()).strip()
        if paragraph:
            paragraphs.append(paragraph)
    return "\n\n".join(paragraphs)


def _extract_pptx(data: bytes) -> str:
    with _zip_archive(data) as archive:
        # Slide files sort lexically, which puts slide10 before slide2; sort on
        # the trailing number so the extracted deck keeps presentation order.
        slides = sorted(
            (name for name in archive.namelist() if name.startswith("ppt/slides/slide")),
            key=lambda name: int("".join(c for c in Path(name).stem if c.isdigit()) or 0),
        )
        blocks: list[str] = []
        for name in slides:
            root = _parse_xml(archive.read(name))
            lines = [
                text.strip()
                for node in root.iter(f"{{{_OOXML_NS['a']}}}p")
                if (text := "".join(node.itertext()).strip())
            ]
            if lines:
                blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _extract_odt(data: bytes) -> str:
    with _zip_archive(data) as archive:
        try:
            xml_data = archive.read("content.xml")
        except KeyError as exc:
            raise UnsupportedDocument("The .odt file has no content.") from exc
    root = _parse_xml(xml_data)
    blocks: list[str] = []
    for tag in ("p", "h"):
        for node in root.iter(f"{{{_ODF_TEXT_NS}}}{tag}"):
            text = "".join(node.itertext()).strip()
            if text:
                blocks.append(text)
    return "\n\n".join(blocks)


def _extract_epub(data: bytes) -> str:
    with _zip_archive(data) as archive:
        documents = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith((".xhtml", ".html", ".htm"))
        )
        blocks: list[str] = []
        for name in documents:
            soup = BeautifulSoup(archive.read(name), "lxml")
            for tag in soup(["script", "style"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)
            if text:
                blocks.append(text)
    return "\n\n".join(blocks)


def _extract_xlsx(data: bytes) -> str:
    from openpyxl import load_workbook

    try:
        workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except (zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise UnsupportedDocument("The spreadsheet could not be read.") from exc
    blocks: list[str] = []
    try:
        for sheet in workbook.worksheets:
            rows = []
            for row in sheet.iter_rows(values_only=True):
                cells = [str(cell).strip() for cell in row if cell is not None]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append(f"# {sheet.title}\n" + "\n".join(rows))
    finally:
        workbook.close()
    return "\n\n".join(blocks)


def _parse_xml(payload: bytes):
    try:
        return ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise UnsupportedDocument("The document markup could not be parsed.") from exc


_BINARY_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".pptx": _extract_pptx,
    ".odt": _extract_odt,
    ".epub": _extract_epub,
    ".xlsx": _extract_xlsx,
}

SUPPORTED_EXTENSIONS = frozenset(TEXT_EXTENSIONS | _BINARY_EXTRACTORS.keys())


def extract_text(data: bytes, filename: str, content_type: str | None = None) -> tuple[str, str]:
    """Return ``(text, source_format)`` for an uploaded file.

    The extension decides the parser; ``content_type`` only fills in when the
    filename has none. Raises ``UnsupportedDocument`` for unknown formats,
    unreadable files, and files that contain no extractable text.
    """
    extension = Path(filename).suffix.lower()
    if not extension and content_type:
        extension = MIME_EXTENSIONS.get(content_type.split(";")[0].strip(), "")

    extractor = _BINARY_EXTRACTORS.get(extension)
    if extractor is not None:
        text = extractor(data)
        source_format = extension.lstrip(".")
    elif extension in TEXT_EXTENSIONS or (not extension and not _is_probable_binary(data)):
        text = data.decode("utf-8", errors="replace")
        source_format = extension.lstrip(".") or "text"
    else:
        raise UnsupportedDocument(f"The file format '{extension or 'unknown'}' is not supported.")

    if not text.strip():
        raise UnsupportedDocument("The file contains no extractable text.")
    return text, source_format
