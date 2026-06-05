"""Tests for the new chunking strategies."""

from pathlib import Path

from rag.models import Document
from chunking.fixed_window import FixedWindowChunker
from chunking.markdown_chunker import MarkdownHeadingChunker
from chunking.page_chunker import PageMarkerChunker


def _doc(text: str) -> Document:
    return Document(id="doc1", path=Path("doc1.txt"), text=text, metadata={})


def test_fixed_window_chunker_splits_text():
    text = "a" * 1200
    doc = _doc(text)
    chunker = FixedWindowChunker(size=500, overlap=50)

    chunks = chunker.chunk(doc)

    assert len(chunks) > 1
    assert chunks[0].text
    assert chunks[0].metadata["strategy"] == "fixed_window"


def test_markdown_heading_chunker_uses_headings():
    text = "# Title\npara1\n\n## Section\npara2"
    doc = _doc(text)
    chunker = MarkdownHeadingChunker()

    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    headings = {c.metadata.get("heading") for c in chunks}
    assert "Title" in headings or "Section" in headings


def test_page_marker_chunker_splits_on_pages():
    text = "[PAGE 1]\nhello\n[PAGE 2]\nworld"
    doc = _doc(text)
    chunker = PageMarkerChunker()

    chunks = chunker.chunk(doc)

    assert len(chunks) >= 2
    pages = {c.metadata.get("page") for c in chunks}
    assert 1 in pages or 2 in pages


