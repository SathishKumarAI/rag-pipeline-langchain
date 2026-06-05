"""
Utility modules for the RAG pipeline.

This package provides reusable utilities for:
- Configuration management
- Logging
- File I/O
- Input validation
- LLM interactions
- Embeddings
- RAG preprocessing
- Typed errors and retry/resilience helpers

Public symbols are imported lazily (PEP 562) so importing one utility (e.g.
``utils.exceptions``) does not eagerly pull in optional heavy dependencies of
the others (pandas, structlog, openai, ...).
"""

# symbol -> defining submodule
_LAZY = {
    "Config": ".config",
    "get_config": ".config",
    "get_logger": ".logging_utils",
    "setup_logging": ".logging_utils",
    "read_text": ".io_utils",
    "write_text": ".io_utils",
    "read_json": ".io_utils",
    "write_json": ".io_utils",
    "safe_write": ".io_utils",
    "validate_document": ".validation",
    "validate_chunk": ".validation",
    "validate_query": ".validation",
    "call_llm": ".llm_utils",
    "format_prompt": ".llm_utils",
    "retry_with_backoff": ".llm_utils",
    "generate_embeddings": ".embeddings",
    "batch_embeddings": ".embeddings",
    "normalize_embeddings": ".embeddings",
    "clean_text": ".rag_preprocessing",
    "chunk_text": ".rag_preprocessing",
    "enrich_metadata": ".rag_preprocessing",
    # error handling
    "RAGError": ".exceptions",
    "ConfigError": ".exceptions",
    "LoaderError": ".exceptions",
    "EmbeddingError": ".exceptions",
    "VectorStoreError": ".exceptions",
    "RetrievalError": ".exceptions",
    "InferenceError": ".exceptions",
    "retry_on_transient": ".resilience",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    mod = import_module(module, __name__)
    return getattr(mod, name)


def __dir__():
    return sorted(__all__)
