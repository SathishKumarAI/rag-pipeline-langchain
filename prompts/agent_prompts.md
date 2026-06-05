# Agent Prompts (from this build) — annotated

The real subagent prompts used while completing this repo, each dissected with
the six W-questions (see [`prompt_engineering_guide.md`](prompt_engineering_guide.md))
and followed by a concrete improvement. These are *task* prompts: instructions
to an AI worker, as opposed to the *product* prompts in
[`rag_system_prompts.md`](rag_system_prompts.md).

---

## Prompt A — Import-failure audit

### The prompt (as sent, condensed)

> Audit the RAG codebase at `/…/RAG` for modules that fail to import.
> **Method:** use the venv python `/…/.venv/bin/python` with
> `PYTHONPATH=/…/pipeline/src`; enumerate every `.py` under `pipeline/src`;
> convert each path to a dotted module; attempt `import_module`; capture the
> exception type and root-cause line.
> **Classify** failures into: (A) missing third-party dependency — list the
> distinct packages; (B) broken relative import / packaging bug; (C) syntax/other.
> **Return** a table `module → class → one-line cause`, the distinct missing
> packages, and the class-B/C bugs called out separately because those are what
> we must fix.
> **Guardrail:** do not fix anything; report only.

### Six-W analysis

- **What** — a classified table of import failures + a list of missing packages.
- **Why** — to separate *environment* issues (just `pip install`) from *real code
  bugs* (must edit). This is the single most useful framing; it turned 14
  failures into "4 deps to install + a handful of bugs to fix".
- **Who** — implicitly a read-only auditor (made explicit by the guardrail).
- **When** — n/a (no temporal scope needed).
- **Where** — exact repo path, venv interpreter, and `PYTHONPATH` were supplied,
  so the agent reproduced the *real* runtime instead of guessing.
- **How** — a concrete algorithm (path→module→import→capture) *and* an output
  contract (the table + the lists) *and* a guardrail (report, don't fix).

### Why it worked

It is reproducible (exact environment), it has a crisp deliverable, and the
"classify A vs B/C" instruction does the analytical work up front so the result
is immediately actionable.

### How to improve it

1. **Force a schema.** Ask for the table as JSON
   (`[{module, class, cause}]`) so the result is machine-mergeable rather than
   prose to re-parse.
2. **Bound the run.** "If a single import hangs > 10s, kill it and mark it
   `TIMEOUT`" — protect against a module with import-time network/IO.
3. **Ask for a fix *hypothesis*, not a fix.** "For class-B/C, suggest the
   one-line change but do not apply it." Keeps it read-only yet speeds the
   follow-up.

---

## Prompt B — Web research: RAG best practices + papers

### The prompt (as sent, condensed)

> Research current (2024–2026) best practices for production RAG and the key
> papers behind them. For each of 8 areas (foundational RAG, chunking, hybrid
> retrieval, re-ranking, query transformation, advanced patterns, evaluation,
> embedding models) give: the core idea in 2–3 sentences, the canonical paper(s)
> with title + authors + venue + year + arXiv/DOI URL, and a well-known OSS
> library that implements it. Verify each URL resolves. Return a markdown report
> organised by the 8 areas plus a final table mapping technique → OSS → whether
> this repo has a slot for it. Cite every claim; do not fabricate titles or URLs
> — if unsure, say so.

### Six-W analysis

- **What** — a structured, citable reference across 8 named areas + a mapping
  table.
- **Why** — to ground the repo's roadmap in real literature (so "next steps" are
  evidence-based, not vibes).
- **Who** — implicitly a research analyst; the "do not fabricate" line sets the
  rigor bar.
- **When** — explicitly "2024–2026" and "current", anchoring against stale facts.
- **Where** — the output's destination (`docs/research/`) and the repo's module
  list were given, so the mapping table could be concrete.
- **How** — enumerated the 8 areas (decomposition), fixed the per-item shape
  (idea + paper + URL + OSS), required URL verification, and demanded explicit
  uncertainty.

### Why it worked

Naming the 8 areas prevents drift and guarantees coverage. Requiring
title+venue+year+URL *and* "say so if unsure" is what kept the output honest —
the agent flagged two papers as "verify" and correctly refused to invent papers
for library-only patterns (recursive/semantic chunking).

### How to improve it

1. **Demand a confidence tag per citation** (`verified` / `unverified`) as a
   column, so downstream readers see provenance at a glance. (The agent did this
   in prose; a column is better.)
2. **Cap source age explicitly** ("prefer sources dated ≥ 2023; flag older").
3. **Ask for a one-line "applies to this repo because…"** on each technique, to
   tighten the mapping table.

---

## Meta-lessons from these two prompts

1. **Decompose, then fan out.** Big questions ("what's wrong?", "what's
   state-of-the-art?") become reliable when split into enumerated sub-parts with
   a per-part output shape.
2. **Give the model the real environment.** Exact paths, the venv, the
   `PYTHONPATH` — the audit only worked because it reproduced the true runtime.
3. **Separate observation from action.** "Report, do not fix" produced a clean
   findings list that the main thread could act on deliberately.
4. **Make honesty cheap.** "If unsure, say so; do not fabricate" plus a request
   for explicit caveats is what makes research output trustworthy.
5. **Prefer schemas over prose** when the output feeds another step.
