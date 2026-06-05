# ADR-0002: Typed error hierarchy + tenacity retry

**Status:** Accepted (2026-06)

## Context

`docs/gaps_and_todo.md` lists "Error Handling" as technical debt: modules raised
bare `Exception`/`ValueError`/`ImportError`, and retry logic was ad-hoc. A
hand-rolled `retry_with_backoff` existed in `utils/llm_utils.py`, and that same
decorator dereferenced `openai.OpenAIError` at **import time**, crashing the
whole module with `AttributeError: 'NoneType' object has no attribute
'OpenAIError'` whenever the optional `openai` package was absent.

## Decision

1. **Typed exception hierarchy** (`utils/exceptions.py`): a `RAGError` base with
   stage-specific subclasses — `ConfigError`, `LoaderError`, `EmbeddingError`,
   `VectorStoreError`, `RetrievalError`, `InferenceError`. Callers can catch by
   stage instead of by bare `Exception`.
2. **Retry via tenacity** (`utils/resilience.py`): a `retry_on_transient`
   decorator built on the open-source `tenacity` library (Apache-2.0) —
   exponential backoff + jitter come from the library, not from us. It
   supersedes the hand-rolled `retry_with_backoff`.
3. **Apply at the network boundary**: the OpenAI embedding calls retry transient
   provider errors (`APIConnectionError`, `APITimeoutError`, `RateLimitError`,
   `InternalServerError`) and wrap non-transient failures as `EmbeddingError`.
4. **Fix the import-time crash**: guard the `llm_utils` decorator so it falls
   back to `(Exception,)` when `openai is None`.

## Consequences

- **+** Consistent, catchable error types across stages; resilient external
  calls with battle-tested backoff.
- **+** `utils.llm_utils` imports cleanly without `openai` installed.
- **+** OSS reused and credited; no bespoke backoff to maintain.
- **−** Two retry paths coexist during migration (`retry_on_transient` and the
  legacy `retry_with_backoff`). Follow-up: migrate remaining call sites and
  delete the legacy decorator.
- **−** Domain exceptions are not yet threaded through every module (loaders,
  retriever, inference still raise some generic errors). Incremental adoption.
