"""
FAISS-backed vector store for the RAG pipeline.

WHAT: Concrete VectorStore over Facebook AI Similarity Search (FAISS).
WHY:  FAISS is a battle-tested, in-process ANN library — we adapt it rather
      than reimplement nearest-neighbour search ourselves.
HOW:  A normalized IndexFlatIP wrapped in IndexIDMap2 gives exact cosine
      search with stable integer ids. FAISS stores ONLY vectors, so text and
      metadata live in a side dict keyed by the same id; both are persisted.

Open-source credit:
    faiss-cpu (Facebook AI Research) — MIT License.
    https://github.com/facebookresearch/faiss
    We use: IndexFlatIP, IndexIDMap2, normalize_L2, write_index/read_index.
    This module is a thin adapter; all similarity-search work is FAISS's.

Usage:
    store = FAISSVectorStore(dim=384, index_path="./faiss_index")
    store.add_documents([{ "text": "...", "embedding": [...], "metadata": {...} }])
    hits = store.search(query_embedding, top_k=5, filter={"source": "a.pdf"})
    store.save()            # persist to index_path
    store = FAISSVectorStore.load("./faiss_index")
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from .vector_store import VectorStore


class FAISSVectorStore(VectorStore):
    """Exact cosine-similarity vector store backed by FAISS.

    Args:
        dim: Embedding dimensionality. Optional — inferred from the first
            batch added if not given.
        index_path: Default location for save()/load(). Optional.

    Notes:
        - Cosine similarity is achieved by L2-normalizing every vector and
          using inner-product search (IndexFlatIP). Scores are in [-1, 1].
        - FAISS has no metadata filtering. `filter` is applied as a Python
          post-filter after over-fetching, so heavy filters may need a larger
          top_k internally (handled automatically up to a cap).
    """

    # Over-fetch multiplier when a metadata filter is supplied, so that the
    # post-filter still has enough candidates to return top_k.
    _FILTER_OVERFETCH = 10

    def __init__(self, dim: Optional[int] = None, index_path: Optional[str] = None, **kwargs: Any):
        try:
            import faiss  # noqa: F401
        except ImportError as exc:  # pragma: no cover - env guard
            raise ImportError(
                "FAISSVectorStore requires 'faiss-cpu'. Install with: pip install faiss-cpu"
            ) from exc

        self._faiss = faiss
        self.dim = dim
        self.index_path = index_path
        self._index = None  # lazily built once dim is known
        # id bookkeeping: FAISS needs int64 ids; the pipeline uses string ids.
        self._next_int_id = 0
        self._str_to_int: Dict[str, int] = {}
        self._int_to_doc: Dict[int, Dict[str, Any]] = {}

        if dim is not None:
            self._build_index(dim)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _build_index(self, dim: int) -> None:
        base = self._faiss.IndexFlatIP(dim)
        self._index = self._faiss.IndexIDMap2(base)
        self.dim = dim

    def _to_matrix(self, vectors: List[List[float]]) -> "Any":
        import numpy as np

        mat = np.asarray(vectors, dtype="float32")
        if mat.ndim == 1:
            mat = mat.reshape(1, -1)
        # cosine == inner product on L2-normalized vectors
        self._faiss.normalize_L2(mat)
        return mat

    # ------------------------------------------------------------------ #
    # VectorStore interface
    # ------------------------------------------------------------------ #
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        if not documents:
            return
        import numpy as np

        if self._index is None:
            self._build_index(len(documents[0]["embedding"]))

        vectors = [doc["embedding"] for doc in documents]
        mat = self._to_matrix(vectors)

        int_ids = []
        for doc in documents:
            str_id = str(doc.get("id", self._next_int_id))
            # reuse the mapping if this string id already exists (upsert)
            if str_id in self._str_to_int:
                int_id = self._str_to_int[str_id]
                self._index.remove_ids(np.asarray([int_id], dtype="int64"))
            else:
                int_id = self._next_int_id
                self._next_int_id += 1
                self._str_to_int[str_id] = int_id
            int_ids.append(int_id)
            self._int_to_doc[int_id] = {
                "id": str_id,
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}),
            }

        self._index.add_with_ids(mat, np.asarray(int_ids, dtype="int64"))

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self._index is None or self._index.ntotal == 0:
            return []

        mat = self._to_matrix([query_embedding])
        fetch = top_k * self._FILTER_OVERFETCH if filter else top_k
        fetch = min(fetch, self._index.ntotal)

        scores, ids = self._index.search(mat, fetch)
        results: List[Dict[str, Any]] = []
        for score, int_id in zip(scores[0], ids[0]):
            if int_id == -1:  # FAISS pads short result sets with -1
                continue
            doc = self._int_to_doc.get(int(int_id))
            if doc is None:
                continue
            if filter:
                meta = doc.get("metadata", {})
                if not all(meta.get(k) == v for k, v in filter.items()):
                    continue
            results.append(
                {"text": doc["text"], "metadata": doc["metadata"], "score": float(score)}
            )
            if len(results) >= top_k:
                break
        return results

    def delete(self, ids: List[str]) -> None:
        if self._index is None:
            return
        import numpy as np

        int_ids = [self._str_to_int[s] for s in ids if s in self._str_to_int]
        if not int_ids:
            return
        self._index.remove_ids(np.asarray(int_ids, dtype="int64"))
        for s in ids:
            int_id = self._str_to_int.pop(s, None)
            if int_id is not None:
                self._int_to_doc.pop(int_id, None)

    def clear(self) -> None:
        self._index = None if self.dim is None else None
        if self.dim is not None:
            self._build_index(self.dim)
        self._next_int_id = 0
        self._str_to_int = {}
        self._int_to_doc = {}

    # ------------------------------------------------------------------ #
    # persistence
    # ------------------------------------------------------------------ #
    def save(self, index_path: Optional[str] = None) -> None:
        path = Path(index_path or self.index_path or "")
        if not str(path):
            raise ValueError("No index_path provided to save()")
        path.mkdir(parents=True, exist_ok=True)
        if self._index is not None:
            self._faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "docstore.pkl", "wb") as fh:
            pickle.dump(
                {
                    "dim": self.dim,
                    "next_int_id": self._next_int_id,
                    "str_to_int": self._str_to_int,
                    "int_to_doc": self._int_to_doc,
                },
                fh,
            )
        # human-readable manifest
        with open(path / "manifest.json", "w") as fh:
            json.dump({"dim": self.dim, "count": len(self._int_to_doc)}, fh, indent=2)

    @classmethod
    def load(cls, index_path: str) -> "FAISSVectorStore":
        path = Path(index_path)
        store = cls(index_path=index_path)
        with open(path / "docstore.pkl", "rb") as fh:
            state = pickle.load(fh)
        store.dim = state["dim"]
        store._next_int_id = state["next_int_id"]
        store._str_to_int = state["str_to_int"]
        store._int_to_doc = state["int_to_doc"]
        faiss_file = path / "index.faiss"
        if faiss_file.exists():
            store._index = store._faiss.read_index(str(faiss_file))
        elif store.dim is not None:
            store._build_index(store.dim)
        return store
