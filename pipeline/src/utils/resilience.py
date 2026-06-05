"""
Retry / resilience helpers for the RAG pipeline.

WHAT: A reusable ``retry_on_transient`` decorator for flaky external calls
      (embedding APIs, vector DB network ops, LLM inference).
WHY:  Transient failures (timeouts, rate limits, dropped connections) should be
      retried with exponential backoff instead of bubbling up on the first hit.
HOW:  Thin wrapper over the ``tenacity`` retry library — we do NOT reimplement
      backoff/jitter ourselves.

Open-source credit:
    tenacity (Julien Danjou & contributors) — Apache License 2.0.
    https://github.com/jd/tenacity
    We use: retry(), stop_after_attempt, wait_exponential, retry_if_exception_type.
"""

from __future__ import annotations

from typing import Callable, Tuple, Type

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Default transient errors worth retrying. Provider SDKs (e.g. openai) raise
# subclasses of these for network/timeout issues.
DEFAULT_TRANSIENT: Tuple[Type[BaseException], ...] = (
    ConnectionError,
    TimeoutError,
)


def retry_on_transient(
    max_attempts: int = 3,
    initial_wait: float = 0.5,
    max_wait: float = 8.0,
    exceptions: Tuple[Type[BaseException], ...] = DEFAULT_TRANSIENT,
) -> Callable:
    """Decorator: retry a function on transient errors with exponential backoff.

    Args:
        max_attempts: Total attempts before giving up (re-raises the last error).
        initial_wait: First backoff in seconds; doubles each retry up to max_wait.
        max_wait: Backoff ceiling in seconds.
        exceptions: Exception types treated as transient/retryable.

    Example:
        @retry_on_transient(max_attempts=5)
        def call_api(): ...
    """
    return retry(
        reraise=True,
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=initial_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
    )


__all__ = ["retry_on_transient", "DEFAULT_TRANSIENT"]
