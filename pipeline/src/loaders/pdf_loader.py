"""Loader for PDF files."""

from __future__ import annotations

from pathlib import Path
from typing import List

from pypdf import PdfReader

from rag.models import Document

from .base import DocumentLoader, register_loader


@register_loader([".pdf"])
class PDFLoader(DocumentLoader):
    """Load text from PDF files as a single Document with page markers.

    This keeps things simple for now: we concatenate all pages into one
    string, but clearly mark page boundaries so later chunkers can build
    page-aware chunks if needed.
    """

    def load(self, path: Path) -> List[Document]:
        reader = PdfReader(str(path))
        pages: List[str] = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            pages.append(f"[PAGE {i + 1}]\n{text}")

        full_text = "\n\n".join(pages)
        doc = Document(
            id=str(path),
            path=path,
            text=full_text,
            metadata={
                "source": "pdf",
                "num_pages": len(reader.pages),
                "suffix": ".pdf",
            },
        )
        return [doc]


