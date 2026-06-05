"""
Typed exception hierarchy for the RAG pipeline.

WHAT: A small tree of exceptions so callers can catch failures by stage
      (loading, embedding, storage, retrieval, inference) instead of bare
      ``Exception``.
WHY:  Standardized error types make retry/fallback logic and user-facing error
      messages consistent across the codebase.
HOW:  Everything derives from ``RAGError``; each stage has its own subclass.

These are intentionally project-specific (no open-source library defines our
domain errors). For *retry* behaviour we reuse the ``tenacity`` library rather
than hand-rolling backoff — see ``utils.resilience``.
"""

from __future__ import annotations


class RAGError(Exception):
    """Base class for all RAG pipeline errors."""


class ConfigError(RAGError):
    """Invalid or missing configuration."""


class LoaderError(RAGError):
    """A document loader failed to read or parse a source."""


class EmbeddingError(RAGError):
    """Embedding generation failed (provider, network, or dimension)."""


class VectorStoreError(RAGError):
    """A vector store operation (add/search/delete/persist) failed."""


class RetrievalError(RAGError):
    """Retrieval failed to return usable candidates."""


class InferenceError(RAGError):
    """LLM inference / answer generation failed."""


__all__ = [
    "RAGError",
    "ConfigError",
    "LoaderError",
    "EmbeddingError",
    "VectorStoreError",
    "RetrievalError",
    "InferenceError",
]
