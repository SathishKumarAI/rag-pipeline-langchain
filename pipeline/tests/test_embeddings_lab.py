"""Smoke tests for embedding wrappers.

These tests intentionally avoid calling real external services when the
corresponding libraries are not installed.
"""

import pytest

from utils import embeddings as emb_utils
from embeddings.openai_embeddings import OpenAIEmbedder
from embeddings.local_embeddings import LocalSentenceTransformerEmbedder


@pytest.mark.skipif(emb_utils.openai is None, reason="openai package not installed")
def test_openai_embedder_constructs():
    embedder = OpenAIEmbedder()
    assert embedder.model_name


@pytest.mark.skipif(
    emb_utils.SentenceTransformer is None,
    reason="sentence-transformers not installed",
)
def test_local_sentence_transformer_embedder_constructs():
    embedder = LocalSentenceTransformerEmbedder()
    assert embedder.model_name


