"""Tests for retry/resilience helpers and the typed exception hierarchy."""

import pytest

from utils.exceptions import EmbeddingError, RAGError, VectorStoreError
from utils.resilience import retry_on_transient


def test_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    @retry_on_transient(max_attempts=3, initial_wait=0.0, max_wait=0.0)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_retry_reraises_after_exhaustion():
    calls = {"n": 0}

    @retry_on_transient(max_attempts=2, initial_wait=0.0, max_wait=0.0)
    def always_fail():
        calls["n"] += 1
        raise TimeoutError("nope")

    with pytest.raises(TimeoutError):
        always_fail()
    assert calls["n"] == 2


def test_retry_does_not_retry_non_transient():
    calls = {"n": 0}

    @retry_on_transient(max_attempts=3, initial_wait=0.0, max_wait=0.0)
    def value_error():
        calls["n"] += 1
        raise ValueError("not transient")

    with pytest.raises(ValueError):
        value_error()
    assert calls["n"] == 1  # tried once, not retried


def test_exception_hierarchy():
    assert issubclass(EmbeddingError, RAGError)
    assert issubclass(VectorStoreError, RAGError)


def test_openai_embeddings_missing_package_raises_typed_error():
    # In this environment openai is not installed, so the typed error fires.
    from utils import embeddings

    if embeddings.openai is not None:
        pytest.skip("openai installed; missing-package path not exercised")
    with pytest.raises(EmbeddingError):
        embeddings.generate_embeddings("hi", provider="openai")
