"""Tests for the new multi-format loaders used in the RAG lab."""

from pathlib import Path

from rag.models import Document
from loaders.base import get_loader_for_path


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def test_text_loader_roundtrip(tmp_path):
    path = _write(tmp_path, "note.txt", "hello world")
    loader = get_loader_for_path(path)
    docs = loader.load(path)

    assert len(docs) == 1
    assert isinstance(docs[0], Document)
    assert "hello world" in docs[0].text


def test_json_loader_basic(tmp_path):
    path = _write(tmp_path, "data.json", '{"a": 1, "b": "x"}')
    loader = get_loader_for_path(path)
    docs = loader.load(path)

    assert len(docs) == 1
    assert "a:" in docs[0].text
    assert "b:" in docs[0].text


def test_csv_loader_basic(tmp_path):
    path = _write(tmp_path, "table.csv", "col1,col2\n1,2\n3,4\n")
    loader = get_loader_for_path(path)
    docs = loader.load(path)

    assert len(docs) == 1
    assert "col1" in docs[0].text
    assert "1" in docs[0].text


