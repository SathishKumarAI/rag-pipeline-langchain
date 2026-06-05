# ADR-0004: Enforce LF line endings via .gitattributes

**Status:** Accepted (2026-06)

## Context

At the start of this work the tree showed 83 "modified" files with **identical
insertion and deletion counts** (9455/9455) and **zero** content change under
`git diff --ignore-all-space`. The diff was pure CRLF↔LF churn from a previous
commit ("trash, needs some cleaning") on a mixed Windows/Linux workflow. Such
noise hides real changes and makes every feature diff unreadable.

## Decision

Add a repo-root `.gitattributes` that sets `* text=auto eol=lf` plus explicit
`eol=lf` for source/text types and `binary` for images/PDFs/index files.
Renormalize the index (`git add --renormalize .`) and rewrite the working tree
to LF in a single, isolated `chore:` commit *before* any feature work, so
feature diffs contain only substantive changes.

Reference: Git docs, gitattributes `text`/`eol` —
https://git-scm.com/docs/gitattributes

## Consequences

- **+** Deterministic LF line endings regardless of contributor OS.
- **+** Feature branches in this effort show real changes only.
- **−** Contributors with `core.autocrlf=true` on Windows still get CRLF in their
  working tree (intended) but commit LF (enforced) — expected Git behaviour.
- **−** The one-time normalization commit touches many files; isolating it keeps
  history bisectable.
