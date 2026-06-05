# Continue Here — resume guide

> Snapshot to pick this work back up in a fresh session. Last updated 2026-06-05.
> Pairs with [`BUILD_LOG.md`](BUILD_LOG.md) (the full reasoning history).

## TL;DR state

- RAG repo went from **uncollectable test suite** → **50 passed, 2 skipped, 0 failed**.
- High-priority gaps closed: real FAISS + Chroma stores, typed errors + retry,
  packaging/import repairs.
- All work is on **stacked feature branches** (each branched off the previous);
  nothing pushed or merged yet.

## Environment (reproduce exactly)

- Python **3.14** on Arch (PEP 668 — no system pip; must use the venv).
- Virtualenv at repo root: `/home/deva/Documents/work/RAG/.venv`
- Installed in venv: `numpy pandas pytest faiss-cpu chromadb tenacity structlog pypdf`
- **Run tests** (no PYTHONPATH needed — set in `pipeline/pytest.ini`):
  ```bash
  cd pipeline && ../.venv/bin/python -m pytest -q
  ```
- Local git identity set in-repo: `Sathish Kumar <sathishkumar786.ml@gmail.com>`.

## Branch / commit trail (stacked, oldest → newest)

| Branch | Commit | Content |
|---|---|---|
| `chore/normalize-line-endings` | `2ab7432` | `.gitattributes` + LF renormalize |
| `feat/vector-stores` | `227e7df` | FAISS + Chroma adapters + tests |
| `feat/error-handling` | `1089a30` | typed errors + tenacity retry |
| `fix/imports-and-packaging` | `37454ca` | import/packaging/test repairs |
| `docs/architecture-research-prompts` | `b84ec17` + this doc | ADRs, BUILD_LOG, research, prompts |

`docs/architecture-research-prompts` HEAD contains **all** the work (stacked).
Base branch for PRs: `main`. Pre-existing work branch: `feature/rag-refactor-llm-platform`.

## Done

- [x] Line endings normalized (ADR-0004)
- [x] FAISS vector store — `pipeline/src/rag/storage/faiss_store.py` (ADR-0001)
- [x] Chroma vector store — `pipeline/src/rag/storage/chroma_store.py` (ADR-0001)
- [x] Typed errors `utils/exceptions.py` + retry `utils/resilience.py` (ADR-0002)
- [x] Lazy package imports `rag/__init__.py`, `utils/__init__.py` (ADR-0003)
- [x] Import/packaging/test repairs — see BUILD_LOG §4 (0 import failures)
- [x] Docs: ADRs, BUILD_LOG, research survey, prompts collection

## Not done / decisions pending (pick up here)

1. **Merge strategy** — stacked branches are unpushed/unmerged. Decide: squash
   the stack into `feature/rag-refactor-llm-platform`, or open one PR per branch
   onto `main`. (Outward-facing — needs user go-ahead.)
2. **Hybrid retrieval + RRF** *(highest ROI — do this next)* — `rag/retriever.py`
   has a dead `use_keyword_search` flag. Wire `rank_bm25` + Reciprocal Rank
   Fusion. See research §3 and §10.
3. **Pinecone store** — currently an explicit `NotImplementedError` (hosted,
   needs API key). Add `pinecone_store.py` adapter for hosted deploy.
4. **Finish error-handling rollout** — thread typed exceptions through loaders /
   retriever / inference; retire legacy `llm_utils.retry_with_backoff`.
5. **Re-ranker / evaluation quality** — verify `rag/re_ranker.py` uses a real
   `sentence-transformers` CrossEncoder; back `rag/evaluation.py` with RAGAS.
6. **Chunking** — add recursive + semantic splitters to `chunking/`.
7. **2 skipped tests** — need optional models (`sentence-transformers` / `openai`)
   not installed in this env.

## Suggested next-session first move

```bash
# verify still green, then start hybrid retrieval
cd pipeline && ../.venv/bin/python -m pytest -q
git checkout docs/architecture-research-prompts
git checkout -b feat/hybrid-retrieval
# implement BM25 + RRF in src/rag/retriever.py behind use_keyword_search
```

## Gotchas learned

- Two test rootings coexist: some tests `from rag...` (needs `src` on path),
  some `from src.rag...` (needs repo root) — `pytest.ini` `pythonpath = src .`
  covers both.
- Package `__init__` files are lazy (PEP 562); import a leaf module directly to
  avoid pulling optional heavy deps.
- FAISS stores vectors only — text/metadata live in a side docstore; metadata
  filtering is post-filter over an over-fetched set.
- Chroma collection names must be 3–512 chars (`[a-zA-Z0-9._-]`).
