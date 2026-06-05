# ADR-0001: Vector store abstraction + OSS-backed adapters

**Status:** Accepted (2026-06)

## Context

The pipeline defines an abstract `VectorStore` interface (`add_documents`,
`search`, `delete`, `clear`) in `rag/storage/vector_store.py`. Only
`InMemoryVectorStore` was real; `PineconeVectorStore`, `FAISSVectorStore`, and
`ChromaVectorStore` were `NotImplementedError` placeholders. Worse, the
`get_vector_store` factory imported them from modules (`faiss_store`,
`chroma_store`, `pinecone_store`) that did not exist, so requesting any backend
but `in_memory` raised `ImportError`.

Two constraints shaped the design:

1. The `VectorStore` contract is **bring-your-own-vector**: callers pass
   precomputed embeddings to `add_documents` and a precomputed `query_embedding`
   to `search`. Many high-level libraries (LangChain vector stores) instead
   *own* the embedding step, which mismatches this contract.
2. The directive: **do not reinvent the wheel — use open source and credit it.**

## Decision

Implement `FAISSVectorStore` and `ChromaVectorStore` as **thin adapters** over
the open-source libraries `faiss-cpu` (MIT) and `chromadb` (Apache-2.0). The
adapter only maps the project's dict-shaped records to/from each library's
native API; all approximate-nearest-neighbour search and persistence is done by
the library, not by us.

- **FAISS:** normalized `IndexFlatIP` wrapped in `IndexIDMap2` (exact cosine +
  stable integer ids), with a side docstore for text/metadata (FAISS stores only
  vectors) and `write_index`/`read_index` persistence. Metadata filtering is a
  Python post-filter over an over-fetched candidate set (FAISS has no native
  filter).
- **Chroma:** one collection with `hnsw:space=cosine`, using Chroma's native
  `where` metadata filter and persistent/ephemeral clients.

Each module header credits the library (name, license, URL, the specific APIs
used). `pinecone` remains unimplemented but now raises an explicit, informative
`NotImplementedError` (it is a hosted service requiring an API key and network).

## Consequences

- **+** Two production-capable local backends, verified by a shared
  parametrized contract test (FAISS, Chroma, InMemory all pass the same suite).
- **+** No heavy framework dependency; the bring-your-own-vector contract is
  preserved; embeddings stay decoupled from storage.
- **+** Credits are explicit, satisfying the OSS-attribution requirement.
- **−** FAISS metadata filtering is O(over-fetch) rather than native; fine for
  the medium-scale target, but for heavy filters a metadata-aware index (or
  Chroma) is preferable.
- **−** Pinecone still needs an adapter before any hosted deployment.

## Alternatives considered

- **Wrap LangChain vector stores.** Rejected: they manage embeddings internally,
  mismatching the precomputed-vector contract, and add a large dependency for
  little gain at this layer.
- **Hand-roll an ANN index.** Rejected outright — that *is* reinventing the
  wheel; faiss/chromadb are the wheel.
