"""
Vector store abstractions for the RAG pipeline.

Concrete backends:
- InMemoryVectorStore: numpy cosine, for tests/small data.
- FAISSVectorStore: faiss-cpu (MIT) — exact cosine, persistent.
- ChromaVectorStore: chromadb (Apache-2.0) — persistent, native metadata filter.
"""

from .vector_store import VectorStore, InMemoryVectorStore, get_vector_store

__all__ = [
    "VectorStore",
    "InMemoryVectorStore",
    "get_vector_store",
    "FAISSVectorStore",
    "ChromaVectorStore",
]


def __getattr__(name: str):
    # Lazy import so importing the package doesn't require faiss/chromadb.
    if name == "FAISSVectorStore":
        from .faiss_store import FAISSVectorStore

        return FAISSVectorStore
    if name == "ChromaVectorStore":
        from .chroma_store import ChromaVectorStore

        return ChromaVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
