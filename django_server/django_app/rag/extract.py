import io
import typing

import docx
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {"txt", "md", "pdf", "docx"}
PAGE_SEPARATOR = "\n\n"


class UnsupportedFileTypeError(ValueError):
    pass


class ExtractedText(typing.NamedTuple):
    text: str
    # Character offset in `text` where each page starts (PDFs only), so a
    # chunk's char_start can be mapped back to the page it came from.
    page_starts: list[int] | None


def extract_document(filename: str, raw: bytes) -> ExtractedText:
    """Pulls plain text out of an uploaded file. Raises
    UnsupportedFileTypeError for anything not in SUPPORTED_EXTENSIONS —
    callers turn that into a 400 rather than silently ingesting garbage.
    """
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"Unsupported file type: .{extension or '?'}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if extension in ("txt", "md"):
        return ExtractedText(raw.decode("utf-8", errors="replace"), None)

    if extension == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        parts: list[str] = []
        page_starts: list[int] = []
        offset = 0
        for page in reader.pages:
            page_starts.append(offset)
            page_text = page.extract_text() or ""
            parts.append(page_text)
            offset += len(page_text) + len(PAGE_SEPARATOR)
        return ExtractedText(PAGE_SEPARATOR.join(parts), page_starts)

    document = docx.Document(io.BytesIO(raw))
    return ExtractedText("\n\n".join(p.text for p in document.paragraphs if p.text), None)


def extract_text(filename: str, raw: bytes) -> str:
    return extract_document(filename, raw).text
