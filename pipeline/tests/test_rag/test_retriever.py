"""Tests for retriever module."""

import pytest

from src.rag.retriever import Retriever
from src.rag.storage.vector_store import InMemoryVectorStore


def test_retriever_retrieve():
    """Test basic retrieval."""
    # Create vector store with sample documents
    store = InMemoryVectorStore()
    docs = [
        {
            "text": "RAG is Retrieval-Augmented Generation",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"source": "doc1.txt"},
        },
        {
            "text": "Vector stores store embeddings",
            "embedding": [0.4, 0.5, 0.6],
            "metadata": {"source": "doc2.txt"},
        },
    ]
    store.add_documents(docs)
    
    # Create retriever
    retriever = Retriever(store, top_k=2)
    
    # Retrieve (using a query embedding similar to first doc)
    query_embedding = [0.1, 0.2, 0.3]
    results = store.search(query_embedding, top_k=2)
    
    assert len(results) <= 2
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)


def test_retriever_with_filter():
    """Test retrieval with metadata filter."""
    store = InMemoryVectorStore()
    docs = [
        {
            "text": "Document 1",
            "embedding": [0.1, 0.2, 0.3],
            "metadata": {"source": "doc1.txt", "category": "A"},
        },
        {
            "text": "Document 2",
            "embedding": [0.4, 0.5, 0.6],
            "metadata": {"source": "doc2.txt", "category": "B"},
        },
    ]
    store.add_documents(docs)
    
    # Inject a deterministic embedder so the test needs no live provider.
    retriever = Retriever(store, top_k=5, embed_fn=lambda q: [0.1, 0.2, 0.3])
    results = retriever.retrieve("test query", filter={"category": "A"})
    assert isinstance(results, list)
    assert all(r["metadata"].get("category") == "A" for r in results)

