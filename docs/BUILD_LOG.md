# Build Log — RAG repo completion effort (2026-06)

A chronological, reasoning-first record of how this repository was diagnosed and
completed, written so the decisions (not just the diffs) are reviewable. Each
entry states **what** was observed, **why** it mattered, and **how** it was
resolved. Companion docs: [ADRs](adr/), [research](research/rag_best_practices.md),
[prompts](../prompts/). **To resume work, start at
[`CONTINUE_HERE.md`](CONTINUE_HERE.md).**

---

## 0. Framing

The request — "finish this project all in one go" — was scoped down to something
achievable and verifiable: implement the **High-priority gaps** from
`gaps_and_todo.md`, fix what is actually broken, and document the work like a
solution architect (feature branches, ADRs, research, prompt engineering). The
guiding constraints, set by the user during the work:

- Use **open source**; do not reinvent the wheel; **credit** what is used.
- One **feature branch per feature**.
- Write all docs as Markdown under `docs/`, plus a `prompts/` collection.
- Show the **thinking and the prompts** professionally, to learn from.

## 1. Diagnosis before action

**Observed.** `git status` showed the entire tree modified. **Why it mattered:**
that looked like a huge in-flight change. **How resolved:** measured it —
`git diff --stat` reported 9455 insertions / 9455 deletions and
`git diff --ignore-all-space --stat` reported *nothing*. The "changes" were pure
CRLF↔LF churn, not content. → **ADR-0004**, fixed first so later diffs are
readable.

**Observed.** `vector_store.py` had `NotImplementedError` placeholders, and the
factory imported non-existent modules. **Environment:** Python 3.14 on Arch with
PEP 668 (no system pip); no venv; numpy/pytest/faiss/chromadb absent. **How
resolved:** created `.venv`, installed exactly what each step needed and verified
imports on 3.14 before writing code against them.

## 2. Vector stores (feature 1)

Chose **thin adapters** over `faiss-cpu` and `chromadb` rather than wrapping a
high-level framework, because the project's `VectorStore` contract is
bring-your-own-vector and the libraries *are* the wheel. Reasoning and trade-offs
in **ADR-0001**. A single parametrized contract test runs FAISS, Chroma, and
InMemory through the same add→search→filter→delete→clear path; FAISS adds a
persistence round-trip. 12 tests green.

While wiring this, discovered `rag/__init__.py` eagerly imported `pipelines`,
whose broken relative import made the whole `rag` package unimportable → made
`__init__` lazy (**ADR-0003**).

## 3. Error handling (feature 2)

Added a typed `RAGError` hierarchy and a `tenacity`-based `retry_on_transient`
decorator (OSS, credited), applied to the OpenAI embedding calls; fixed an
import-time crash in `llm_utils` (decorator dereferencing `openai.OpenAIError`
when `openai` is `None`). Reasoning in **ADR-0002**. Making `utils/__init__`
lazy here (same pattern as ADR-0003) stopped `import utils.exceptions` from
dragging in pandas/structlog/openai.

## 4. Agent-driven audit (feature 3)

Ran subagents in parallel to find problems without burning main-thread context:

1. **Import-failure audit** (Explore agent): attempted to import all 89 modules
   under `pipeline/src`, classifying each failure as missing-dependency vs. real
   code bug. Found 14 failing modules.
2. **Web research** (general-purpose agent): verified RAG best practices + papers
   → `docs/research/rag_best_practices.md`.

The exact agent prompts and how to improve them are in
[`prompts/agent_prompts.md`](../prompts/agent_prompts.md).

### Real code defects found and fixed

| Defect | Root cause | Fix |
|---|---|---|
| `rag/*` unimportable | 16 `from ..utils` relative imports "beyond top-level package" given the `src`-on-path layout | → absolute `from utils.*` |
| `vectorstores/faiss_store.py` ImportError | imported `FAISSVectorStore` from `vector_store` (it lives in `faiss_store`) | corrected import |
| `utils/llm_utils` crash on import | decorator used `openai.OpenAIError` with `openai=None` | guarded fallback |
| `PyPDF2` ModuleNotFound | code imported `PyPDF2`; requirements pin `pypdf` | → `pypdf` |
| 4 lab tests uncollectable | imported `rag_pipeline.{embeddings,loaders,chunking}.*` (live at top level) | matched `lab_pipeline.py`'s own imports |
| `No loader registered for .txt` / 0 chunks | `loaders/__init__` never imported the concrete loaders, so `@register_loader` never ran | import loaders (optional ones defensively) |
| retriever untestable without a live API | embedding provider hardcoded | added injectable `embed_fn` (DI) |

Result: **0 import failures** (was 14); the test suite went from *uncollectable*
to **50 passed, 2 skipped**. `pytest.ini` now sets `pythonpath = src .` so the
suite runs with no manual `PYTHONPATH`.

## 5. Verification discipline

Every feature was proven by running tests in the venv before committing; no
feature was marked done on partial/failing state. Counts quoted above are actual
`pytest` output, not estimates.

## 6. Branch / commit trail

| Branch | Commit | What |
|---|---|---|
| `chore/normalize-line-endings` | `chore: normalize line endings to LF` | ADR-0004 |
| `feat/vector-stores` | `feat(storage): real FAISS and Chroma vector stores` | ADR-0001 |
| `feat/error-handling` | `feat(utils): typed error hierarchy + tenacity-based retry` | ADR-0002/0003 |
| `fix/imports-and-packaging` | `fix: repair packaging, imports, and the test suite` | §4 defects |
| `docs/architecture-research-prompts` | this doc + ADRs + research + prompts | docs |

## 7. What remains (honest gaps)

- **Pinecone** backend (hosted; needs API key) — explicit `NotImplementedError`.
- **Hybrid retrieval + RRF** — `use_keyword_search` is still a dead flag; highest-ROI next step (see research §10).
- Domain exceptions not yet threaded through every module; one legacy retry path remains.
- `re_ranker.py` / `evaluation.py` quality (real cross-encoder, RAGAS) not yet verified against the research recommendations.
- 2 skipped tests require optional models (`sentence-transformers`/`openai`) not installed here.
