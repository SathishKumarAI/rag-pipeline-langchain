# Architecture Decision Records (ADRs)

An ADR captures a single architectural decision: its context, the choice made,
and the consequences. They are immutable once accepted — to change a decision,
add a new ADR that supersedes the old one.

Format: Status · Context · Decision · Consequences (after Michael Nygard's
template, https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions).

| ADR | Title | Status |
|---|---|---|
| [0001](ADR-0001-vector-store-abstraction.md) | Vector store abstraction + OSS-backed adapters | Accepted |
| [0002](ADR-0002-error-handling-and-retry.md) | Typed error hierarchy + tenacity retry | Accepted |
| [0003](ADR-0003-lazy-package-imports.md) | Lazy package `__init__` imports (PEP 562) | Accepted |
| [0004](ADR-0004-line-ending-normalization.md) | Enforce LF line endings via .gitattributes | Accepted |
