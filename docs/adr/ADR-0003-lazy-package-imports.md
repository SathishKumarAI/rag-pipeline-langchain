# ADR-0003: Lazy package `__init__` imports (PEP 562)

**Status:** Accepted (2026-06)

## Context

`rag/__init__.py` and `utils/__init__.py` eagerly imported every public symbol
from every submodule. Side effects:

- Importing *one* leaf module pulled in *all* siblings and their optional heavy
  dependencies. `import rag.storage.faiss_store` triggered `rag/__init__` →
  `pipelines` → (a then-broken) `from ..utils...`, and `import utils.exceptions`
  dragged in `pandas`, `structlog`, `openai`. A single missing optional dep made
  the whole package unimportable — which is why the test suite could not even be
  collected.

## Decision

Convert both package `__init__.py` files to **lazy attribute access** using
PEP 562 (`module __getattr__`). Keep a name→submodule map; import the submodule
only when its symbol is first accessed. Public API and `__all__` are unchanged;
`__dir__` is provided for discoverability.

Reference: PEP 562 — https://peps.python.org/pep-0562/

## Consequences

- **+** `import rag.storage.X` no longer imports `pipelines`/`inference`; one
  missing optional dependency no longer breaks unrelated imports.
- **+** Faster, lighter imports for narrow use (e.g. just the vector store).
- **+** Unblocked test collection (combined with the relative-import fixes in
  ADR-adjacent work).
- **−** Import errors surface at first *attribute access* rather than at package
  import — slightly less eager failure. Acceptable trade for decoupling.
- **−** Static analysers / IDEs may need the `_LAZY`/`__all__` hints to resolve
  names; they are kept in sync deliberately.
