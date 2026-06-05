"""FAISS-backed vector store adapter for experiments.

The core layer provides a real FAISS implementation in
`rag.storage.faiss_store`. This module exposes a small adapter class so the
experimentation code can depend on a simple constructor without touching
low-level details.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rag.storage.faiss_store import FAISSVectorStore

from .base import VectorStore


class FAISSStore(FAISSVectorStore, VectorStore):  # type: ignore[misc]
    """Thin adapter around the core FAISSVectorStore (faiss-cpu, MIT)."""

    def __init__(self, index_path: Optional[str] = None, **kwargs: Any) -> None:
        super().__init__(index_path=index_path, **kwargs)

    # Methods are inherited from FAISSVectorStore; we keep the type
    # signatures here for clarity.

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:  # pragma: no cover - delegated
        return super().add_documents(documents)

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:  # pragma: no cover - delegated
        return super().search(query_embedding, top_k=top_k, filter=filter)


