# rag-pipeline-langchain — STATUS

Update this when you STOP working, not when you start.

- **Last touched:** 2026-08-16 (read-only survey; no code changed)
- **Where I stopped:** Abandoned **mid-migration**, with the migration's own docs
  claiming it finished. Nothing here runs.
- **Next action:** none until you decide between the two trees below.
- **Blocked on:** nothing technical. It needs a decision, not work.

## The state, measured

**136 of 230 tracked files are 0 bytes.** Not stubs with a `pass` — empty.

The damage is not spread evenly. It is a **half-finished restructure**, with two
parallel source trees:

| Tree | State |
|---|---|
| `pipeline/src/{loaders,chunking,cleaning,embeddings,rag,utils,tracking,vectorstores}` | **Real.** 46 files with content — `rag/inference.py` (5.7 KB), `rag/evaluation.py` (7.8 KB) |
| `pipeline/src/rag_pipeline/**` | **Hollow.** 38 empty files — ingestion, parsing, chunking, embedding, storage, retrieval, generation, workflows |

So someone began moving the working code into a tidier `rag_pipeline` package, created
every destination file, and stopped before moving any content. Also empty: all of
`pipeline/api/` (5 files, including `fastapi_app.py`), 11 of the 25 files in
`pipeline/tests/`, and the root `Makefile`, `pyproject.toml` and `setup.cfg`. The two
`.github/workflows/` files are 0 bytes too, so **there is no CI even nominally**.

## The trap

`docs/` contains `REFACTORING_COMPLETE.md` and `PR_SUMMARY.md`, both describing the
refactor as done. They are wrong. `docs/gaps_and_todo.md` in the same directory
contradicts them and admits placeholder implementations remain.

Believe the byte counts, not the docs. That is the single most useful thing on this
page: anyone returning here will read `REFACTORING_COMPLETE.md` first and lose an hour.

## Why it is not part of the RAG program

The three sibling apps are local-first, keyless, offline-testable, and share one engine.
This is a cloud platform: `pinecone-client`, `boto3`, `langgraph`, `mlflow`, with CDK /
CloudFormation / Helm / Kustomize manifests under `mlops/` and 309 KB of docs. Its
`requirements.txt` is **completely unpinned**. Running it needs AWS and Pinecone
credentials, so it cannot inherit the no-keys posture the other three have.

## If you come back to it

1. **Salvage** — the populated tree is real work. Lift `pipeline/src/rag/inference.py`
   and `evaluation.py` out, keep the ideas, drop the shell.
2. **Finish the migration** — complete the `rag_pipeline` move, delete the old tree,
   fill in the API and tests. Substantial, and it only pays off if you actually want an
   AWS/Pinecone deployment story that the other three deliberately avoid.
3. **Archive** — `~/coding/archive/`. Most honest option today. Nothing here is
   runnable, and the empty-file skeleton is a liability that reads as progress.

Whichever you pick, **delete or correct `REFACTORING_COMPLETE.md` first.** A document
that overstates completion is worse than no document.
