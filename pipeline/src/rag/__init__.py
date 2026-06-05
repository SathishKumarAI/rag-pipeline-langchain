"""
RAG (Retrieval-Augmented Generation) pipeline modules.

This package provides a complete RAG implementation with:
- Index building from raw documents
- Vector and keyword retrieval
- Re-ranking
- Prompt building
- End-to-end inference
- Evaluation metrics

Public symbols are imported lazily (PEP 562) so that importing a single
submodule (e.g. ``rag.storage.faiss_store``) does not eagerly pull in heavy or
optional dependencies of unrelated submodules.
"""

__all__ = [
    "build_index",
    "run_rag_query",
    "IndexBuilder",
    "Retriever",
    "ReRanker",
    "PromptBuilder",
    "RAGInference",
    "RAGEvaluator",
]

# symbol -> submodule that defines it
_LAZY = {
    "build_index": ".pipelines",
    "run_rag_query": ".pipelines",
    "IndexBuilder": ".index_builder",
    "Retriever": ".retriever",
    "ReRanker": ".re_ranker",
    "PromptBuilder": ".prompt_builder",
    "RAGInference": ".inference",
    "RAGEvaluator": ".evaluation",
}


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(module, __name__)
    return getattr(mod, name)


def __dir__():
    return sorted(__all__)
