from __future__ import annotations

import io

import docx
import pytest

from app.rag.extract import UnsupportedFileTypeError, extract_text


def test_extract_text_from_txt():
    assert extract_text("notes.txt", b"hello world") == "hello world"


def test_extract_text_unsupported_extension_raises():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text("archive.zip", b"whatever")


def test_extract_text_from_docx_round_trip():
    document = docx.Document()
    document.add_paragraph("First paragraph.")
    document.add_paragraph("Second paragraph.")
    buffer = io.BytesIO()
    document.save(buffer)

    text = extract_text("doc.docx", buffer.getvalue())

    assert "First paragraph." in text
    assert "Second paragraph." in text
