"""Tests for the FAISS and Chroma vector store adapters.

These exercise the concrete VectorStore implementations through the same
contract: add -> search (ranking + filter) -> delete -> clear, plus FAISS
persistence round-trip.
"""

import pytest

from rag.storage import get_vector_store
from rag.storage.vector_store import InMemoryVectorStore

faiss = pytest.importorskip("faiss")
chromadb = pytest.importorskip("chromadb")

from rag.storage.faiss_store import FAISSVectorStore  # noqa: E402
from rag.storage.chroma_store import ChromaVectorStore  # noqa: E402


DOCS = [
    {"id": "a", "text": "the cat sat on the mat", "embedding": [1.0, 0.0, 0.0], "metadata": {"src": "x"}},
    {"id": "b", "text": "dogs are loyal animals", "embedding": [0.0, 1.0, 0.0], "metadata": {"src": "y"}},
    {"id": "c", "text": "feline pets like cats", "embedding": [0.9, 0.1, 0.0], "metadata": {"src": "x"}},
]


@pytest.fixture(params=["faiss", "chroma", "memory"])
def store(request):
    """A fresh empty store for each backend, per test (no cross-test state)."""
    if request.param == "faiss":
        return FAISSVectorStore(dim=3)
    if request.param == "chroma":
        return ChromaVectorStore(index_path=None, collection="test_collection")
    return InMemoryVectorStore()


def test_add_and_rank(store):
    store.add_documents(DOCS)
    hits = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(hits) == 2
    # nearest to [1,0,0] is doc "a" (cat), then "c" (feline)
    assert "cat" in hits[0]["text"]
    assert hits[0]["score"] >= hits[1]["score"]


def test_metadata_filter(store):
    store.add_documents(DOCS)
    hits = store.search([0.0, 1.0, 0.0], top_k=5, filter={"src": "y"})
    assert all(h["metadata"].get("src") == "y" for h in hits)
    assert any("dogs" in h["text"] for h in hits)


def test_delete_and_clear(store):
    store.add_documents(DOCS)
    store.delete(["a"])
    hits = store.search([1.0, 0.0, 0.0], top_k=5)
    assert all("the cat sat" not in h["text"] for h in hits)
    store.clear()
    assert store.search([1.0, 0.0, 0.0], top_k=5) == []


def test_faiss_persistence_roundtrip(tmp_path):
    path = str(tmp_path / "idx")
    store = FAISSVectorStore(dim=3, index_path=path)
    store.add_documents(DOCS)
    store.save()

    loaded = FAISSVectorStore.load(path)
    hits = loaded.search([1.0, 0.0, 0.0], top_k=1)
    assert len(hits) == 1
    assert "cat" in hits[0]["text"]


def test_factory_wires_backends():
    assert isinstance(get_vector_store("faiss"), FAISSVectorStore)
    assert isinstance(get_vector_store("chroma"), ChromaVectorStore)
    with pytest.raises(NotImplementedError):
        get_vector_store("pinecone")
    with pytest.raises(ValueError):
        get_vector_store("nope")
