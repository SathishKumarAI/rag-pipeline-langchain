"""
Embedding utilities for the RAG pipeline.

WHAT: Functions to generate embeddings from raw text, batch processing, and normalization.
WHY: Provides a unified interface for embedding generation, handles batching for efficiency,
     and ensures embeddings are properly normalized.
HOW: Abstracts embedding generation behind a simple interface, implements batch processing
     to handle large volumes efficiently, and provides normalization utilities.

Usage:
    from utils.embeddings import generate_embeddings, batch_embeddings, normalize_embeddings
    
    embedding = generate_embeddings("text to embed", model="text-embedding-ada-002")
    embeddings = batch_embeddings(["text1", "text2"], model="text-embedding-ada-002")
    normalized = normalize_embeddings(embedding)
"""

import numpy as np
from typing import List, Optional, Union

from .exceptions import EmbeddingError
from .resilience import DEFAULT_TRANSIENT, retry_on_transient

try:
    import openai
except ImportError:
    openai = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# Transient OpenAI errors worth retrying, merged with the generic set. Built
# only when openai is installed so the import stays optional.
_OPENAI_TRANSIENT = DEFAULT_TRANSIENT
if openai is not None:
    _OPENAI_TRANSIENT = DEFAULT_TRANSIENT + tuple(
        e
        for e in (
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "APITimeoutError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "InternalServerError", None),
        )
        if e is not None
    )


def generate_embeddings(
    text: str,
    model: str = "text-embedding-ada-002",
    provider: str = "openai",
    api_key: Optional[str] = None,
) -> List[float]:
    """
    Generate embeddings for a single text.
    
    Args:
        text: Text to embed
        model: Model name/identifier
        provider: Provider name ("openai", "sentence-transformers", "bedrock")
        api_key: API key (if not provided, uses environment/config)
        
    Returns:
        Embedding vector as list of floats
        
    Raises:
        ValueError: If provider is not supported
        Exception: If embedding generation fails
    """
    if provider == "openai":
        return _generate_openai_embeddings(text, model, api_key)
    elif provider == "sentence-transformers":
        return _generate_sentence_transformer_embeddings(text, model)
    elif provider == "bedrock":
        return _generate_bedrock_embeddings(text, model)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def batch_embeddings(
    texts: List[str],
    model: str = "text-embedding-ada-002",
    provider: str = "openai",
    batch_size: int = 100,
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts.
    
    Args:
        texts: List of texts to embed
        model: Model name/identifier
        provider: Provider name
        batch_size: Number of texts to process in each batch
        api_key: API key (if not provided, uses environment/config)
        
    Returns:
        List of embedding vectors
    """
    if provider == "openai":
        return _batch_openai_embeddings(texts, model, batch_size, api_key)
    elif provider == "sentence-transformers":
        return _batch_sentence_transformer_embeddings(texts, model, batch_size)
    elif provider == "bedrock":
        return _batch_bedrock_embeddings(texts, model, batch_size)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")


def normalize_embeddings(embeddings: Union[List[float], np.ndarray]) -> np.ndarray:
    """
    Normalize embeddings to unit length (L2 normalization).
    
    Args:
        embeddings: Embedding vector or array of embeddings
        
    Returns:
        Normalized embeddings as numpy array
    """
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    
    # Handle 1D (single embedding) or 2D (batch of embeddings)
    if embeddings.ndim == 1:
        norm = np.linalg.norm(embeddings)
        if norm == 0:
            return embeddings
        return embeddings / norm
    else:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return embeddings / norms


def check_embedding_dimension(
    embedding: List[float],
    expected_dim: Optional[int] = None,
) -> int:
    """
    Check embedding dimension and optionally validate against expected dimension.
    
    Args:
        embedding: Embedding vector
        expected_dim: Expected dimension (if provided, validates)
        
    Returns:
        Actual dimension
        
    Raises:
        ValueError: If dimension doesn't match expected
    """
    dim = len(embedding)
    if expected_dim is not None and dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {dim}"
        )
    return dim


@retry_on_transient(exceptions=_OPENAI_TRANSIENT)
def _generate_openai_embeddings(
    text: str,
    model: str,
    api_key: Optional[str],
) -> List[float]:
    """Generate embeddings using OpenAI API (retries transient failures)."""
    if openai is None:
        raise EmbeddingError("openai package not installed")

    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.embeddings.create(
            model=model,
            input=text,
        )
        return response.data[0].embedding
    except _OPENAI_TRANSIENT:
        raise  # let retry_on_transient handle the backoff/retry
    except Exception as exc:  # non-transient -> typed domain error
        raise EmbeddingError(f"OpenAI embedding failed: {exc}") from exc


@retry_on_transient(exceptions=_OPENAI_TRANSIENT)
def _batch_openai_embeddings(
    texts: List[str],
    model: str,
    batch_size: int,
    api_key: Optional[str],
) -> List[List[float]]:
    """Generate embeddings for a batch using OpenAI API (retries transient failures)."""
    if openai is None:
        raise EmbeddingError("openai package not installed")

    try:
        client = openai.OpenAI(api_key=api_key)
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            response = client.embeddings.create(
                model=model,
                input=batch,
            )
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

        return all_embeddings
    except _OPENAI_TRANSIENT:
        raise  # let retry_on_transient handle the backoff/retry
    except Exception as exc:  # non-transient -> typed domain error
        raise EmbeddingError(f"OpenAI batch embedding failed: {exc}") from exc


def _generate_sentence_transformer_embeddings(
    text: str,
    model: str,
) -> List[float]:
    """Generate embeddings using sentence-transformers."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers package not installed")
    
    model_instance = SentenceTransformer(model)
    embedding = model_instance.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def _batch_sentence_transformer_embeddings(
    texts: List[str],
    model: str,
    batch_size: int,
) -> List[List[float]]:
    """Generate embeddings for a batch using sentence-transformers."""
    if SentenceTransformer is None:
        raise ImportError("sentence-transformers package not installed")
    
    model_instance = SentenceTransformer(model)
    embeddings = model_instance.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
    )
    return embeddings.tolist()


def _generate_bedrock_embeddings(
    text: str,
    model: str,
) -> List[float]:
    """Generate embeddings using AWS Bedrock (placeholder)."""
    # This would need to be implemented based on Bedrock embedding models
    raise NotImplementedError("Bedrock embeddings not yet implemented")


def _batch_bedrock_embeddings(
    texts: List[str],
    model: str,
    batch_size: int,
) -> List[List[float]]:
    """Generate embeddings for a batch using AWS Bedrock (placeholder)."""
    raise NotImplementedError("Bedrock embeddings not yet implemented")

