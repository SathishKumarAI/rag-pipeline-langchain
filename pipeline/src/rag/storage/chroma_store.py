"""
Chroma-backed vector store for the RAG pipeline.

WHAT: Concrete VectorStore over Chroma, an open-source embedding database.
WHY:  Chroma already provides persistence, metadata filtering and HNSW ANN
      search out of the box — we adapt it instead of building those ourselves.
HOW:  One Chroma collection holds vectors + documents + metadata. We pass our
      precomputed embeddings straight through (Chroma also supports its own
      embedding functions, but the pipeline brings its own vectors). Cosine
      space is selected via the collection's hnsw:space setting.

Open-source credit:
    chromadb (Chroma) — Apache License 2.0.
    https://github.com/chroma-core/chroma
    We use: PersistentClient/EphemeralClient, collection add/query/delete,
    native `where` metadata filtering, and hnsw:space="cosine".
    This module is a thin adapter; storage and ANN search are Chroma's.

Usage:
    store = ChromaVectorStore(index_path="./chroma_db", collection="rag")
    store.add_documents([{ "text": "...", "embedding": [...], "metadata": {...} }])
    hits = store.search(query_embedding, top_k=5, filter={"source": "a.pdf"})
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """Vector store backed by Chroma.

    Args:
        index_path: Directory for the persistent client. If None, an in-memory
            (ephemeral) client is used — handy for tests.
        collection: Collection name. Defaults to "rag".
    """

    def __init__(
        self,
        index_path: Optional[str] = None,
        collection: str = "rag",
        **kwargs: Any,
    ):
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - env guard
            raise ImportError(
                "ChromaVectorStore requires 'chromadb'. Install with: pip install chromadb"
            ) from exc

        self.index_path = index_path
        self.collection_name = collection
        if index_path:
            self._client = chromadb.PersistentClient(path=index_path)
        else:
            self._client = chromadb.EphemeralClient()
        # cosine space so scores line up with the other stores
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._next_id = self._collection.count()

    # ------------------------------------------------------------------ #
    # VectorStore interface
    # ------------------------------------------------------------------ #
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        ids, embeddings, texts, metadatas = [], [], [], []
        for doc in documents:
            doc_id = str(doc.get("id", self._next_id))
            if "id" not in doc:
                self._next_id += 1
            ids.append(doc_id)
            embeddings.append(doc["embedding"])
            texts.append(doc.get("text", ""))
            # Chroma rejects empty metadata dicts; supply a neutral placeholder.
            meta = doc.get("metadata") or {"_": ""}
            metadatas.append(meta)
        # upsert so re-adding the same id updates rather than duplicates
        self._collection.upsert(
            ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._collection.count() == 0:
            return []
        where = filter or None
        res = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        results: List[Dict[str, Any]] = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            meta = dict(meta or {})
            meta.pop("_", None)  # strip placeholder
            # Chroma returns cosine *distance* (1 - cosine_sim); convert to score
            results.append({"text": text, "metadata": meta, "score": 1.0 - float(dist)})
        return results

    def delete(self, ids: List[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=[str(i) for i in ids])

    def clear(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._next_id = 0
