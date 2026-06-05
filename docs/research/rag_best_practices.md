# Production RAG (2024–2026): Best Practices, Key Papers & Open-Source

> **Purpose.** A verified reference of the techniques, canonical papers, and
> open-source libraries that define modern Retrieval-Augmented Generation, plus
> a mapping of each technique to where it does (or could) live in *this* repo.
>
> **Provenance.** Compiled via web research; every arXiv/DOI link below was
> resolved during research. Where a "best practice" is an engineering pattern
> rather than a single paper, that is stated explicitly — no papers were
> invented. Items flagged *verify* should be re-checked before citing in a
> formal document.

---

## 1. Foundational RAG

| Idea | Paper | Link | OSS |
|---|---|---|---|
| **RAG** — parametric generator + non-parametric retriever, retrieved passages marginalized into generation | *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, Lewis et al., NeurIPS 2020 | https://arxiv.org/abs/2005.11401 | HF `transformers` (`RagSequence/TokenForGeneration`) |
| **REALM** — pre-trains the retriever itself, end-to-end via masked-LM signal | *REALM: Retrieval-Augmented Language Model Pre-Training*, Guu et al., ICML 2020 | https://arxiv.org/abs/2002.08909 | `google-research/language` |
| **FiD** — encode passages independently, fuse evidence in the decoder; scales to many passages | *Leveraging Passage Retrieval with Generative Models for Open Domain QA*, Izacard & Grave, EACL 2021 | https://arxiv.org/abs/2007.01282 | `facebookresearch/FiD` |

## 2. Chunking Strategies

| Idea | Source | Link | OSS |
|---|---|---|---|
| **Fixed-window** — fixed token spans + overlap; baseline, ignores structure | engineering convention (no paper) | — | LangChain `CharacterTextSplitter` |
| **Recursive** — split on a separator priority list (¶ → line → sentence → word) | library pattern (no paper) | — | LangChain `RecursiveCharacterTextSplitter` |
| **Semantic** — boundary where consecutive-sentence embedding distance spikes | practitioner technique (no paper) | — | LlamaIndex `SemanticSplitterNodeParser`, LangChain `SemanticChunker` |
| **Late chunking** — embed full doc with long-context model, then split + pool; ~3.6% avg retrieval gain | *Late Chunking*, Günther et al. (Jina AI), 2024 | https://arxiv.org/abs/2409.04701 | `jina-embeddings-v2/v3` |

## 3. Hybrid Retrieval (dense + sparse, fusion)

| Idea | Source | Link | OSS |
|---|---|---|---|
| **Hybrid dense + BM25** — lexical catches exact/rare terms, dense catches paraphrase | DPR (Karpukhin 2020) + classic BM25; hybrid is an engineering pattern | https://arxiv.org/abs/2004.04906 (DPR) | `rank_bm25`, LangChain `EnsembleRetriever` |
| **Reciprocal Rank Fusion (RRF)** — fuse ranked lists by Σ 1/(k+rank), k≈60; rank-only, avoids score-scale mismatch | *RRF Outperforms Condorcet…*, Cormack et al., SIGIR 2009 | https://dl.acm.org/doi/10.1145/1571941.1572114 · PDF: https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf | built into Elastic/OpenSearch/Weaviate; ~10 LOC |

## 4. Re-ranking

| Idea | Paper | Link | OSS |
|---|---|---|---|
| **Cross-encoder** — joint (query, passage) encode; accurate but costly → re-rank a shortlist | *Sentence-BERT*, Reimers & Gurevych, EMNLP 2019 | https://arxiv.org/abs/1908.10084 | `sentence-transformers` `CrossEncoder` |
| **ColBERT / late interaction** — per-token multi-vectors, MaxSim scoring | *ColBERT*, Khattab & Zaharia, SIGIR 2020 | https://arxiv.org/abs/2004.12832 | `stanford-futuredata/ColBERT`, RAGatouille |
| **monoT5** — seq2seq ranking, "true"/"false" token logit as relevance | *Document Ranking with a Pretrained Seq2Seq Model*, Nogueira et al., EMNLP Findings 2020 | https://arxiv.org/abs/2003.06713 | `castorini/pygaggle` |
| **BGE / Cohere rerankers** — drop-in cross-encoder rerankers | BGE: *C-Pack*, Xiao et al., 2023/24 | https://arxiv.org/abs/2309.07597 | `FlagEmbedding` (`bge-reranker-v2`); Cohere Rerank (hosted) |

## 5. Query Transformation

| Idea | Paper | Link | OSS |
|---|---|---|---|
| **HyDE** — LLM writes a hypothetical answer, embed *that*, retrieve real docs | *Precise Zero-Shot Dense Retrieval without Relevance Labels*, Gao et al., ACL 2023 | https://arxiv.org/abs/2212.10496 | LangChain `HypotheticalDocumentEmbedder` |
| **Multi-query** — generate paraphrases, retrieve each, union | library pattern (no paper) | — | LangChain `MultiQueryRetriever` |
| **Query rewriting** — rewrite-retrieve-read | *Query Rewriting for RAG LLMs*, Ma et al., EMNLP 2023 | https://arxiv.org/abs/2305.14283 *(verify ID)* | LangChain |
| **Step-back** — derive an abstract question, retrieve broader concept, then answer | *Take a Step Back*, Zheng et al. (DeepMind), ICLR 2024 | https://arxiv.org/abs/2310.06117 | LangChain step-back templates |

## 6. Advanced Patterns

| Idea | Paper | Link | OSS |
|---|---|---|---|
| **Self-RAG** — LM decides when to retrieve + self-critiques with reflection tokens | *Self-RAG*, Asai et al., ICLR 2024 | https://arxiv.org/abs/2310.11511 | `AkariAsai/self-rag`; LangGraph |
| **Corrective RAG (CRAG)** — retrieval evaluator → decompose-recompose or web-search fallback | *Corrective Retrieval Augmented Generation*, Yan et al., 2024 | https://arxiv.org/abs/2401.15884 | `HuskyInSalt/CRAG`; LangGraph |
| **RAG-Fusion** — multi-query + RRF | Rackauckas, 2024 *(short paper; mostly a blog-popularized pattern)* | https://arxiv.org/abs/2402.03367 | LangChain (multi-query + RRF) |
| **GraphRAG** — LLM-built entity graph + community summaries for global questions | *From Local to Global: A Graph RAG Approach…*, Edge et al. (MSR), 2024 | https://arxiv.org/abs/2404.16130 | `microsoft/graphrag`; LlamaIndex property graph |

## 7. Evaluation

| Idea | Paper | Link | OSS |
|---|---|---|---|
| **RAGAS** — reference-free LLM eval: faithfulness, answer relevancy, context precision/recall | *RAGAS*, Es et al., EACL 2024 demo | https://arxiv.org/abs/2309.15217 | `explodinggradients/ragas` |
| **ARES** — trained LM judges + prediction-powered inference for statistical guarantees | *ARES*, Saad-Falcon et al., NAACL 2024 | https://arxiv.org/abs/2311.09476 | `stanford-futuredata/ARES` |

Core metrics (faithfulness, answer relevancy, context precision/recall) are also
in TruLens, DeepEval, and Arize Phoenix.

## 8. Embedding Models Worth Knowing

| Model | Source | Link | Access |
|---|---|---|---|
| SBERT | Reimers & Gurevych, EMNLP 2019 | https://arxiv.org/abs/1908.10084 | `sentence-transformers` |
| E5 | Wang et al., 2022 | https://arxiv.org/abs/2212.03533 | `intfloat/e5-*` |
| GTE | Li et al., 2023 | https://arxiv.org/abs/2308.03281 | `thenlper/gte-*`, `Alibaba-NLP/gte-*` |
| BGE (C-Pack) | Xiao et al., 2023/24 | https://arxiv.org/abs/2309.07597 | `BAAI/bge-*`, `FlagEmbedding` |
| OpenAI text-embedding-3 | OpenAI (no paper; large=3072-d, Matryoshka) | https://openai.com/index/new-embedding-models-and-api-updates/ | hosted API |

MTEB (Muennighoff et al.) is the standard leaderboard for choosing among these.

---

## 9. Technique → OSS → this repo's slot

This repo's structure: `pipeline/src/{loaders, chunking, cleaning, embeddings,
rag/{retriever, re_ranker, prompt_builder, evaluation, storage/{faiss_store,
chroma_store}}, vectorstores}`.

| Technique | Recommended OSS | Slot in this repo |
|---|---|---|
| Foundational RAG | HF `transformers` | ✅ `rag/pipelines.py`, `rag/inference.py` |
| Fixed-window chunking | LangChain `CharacterTextSplitter` | ✅ `chunking/fixed_window.py` |
| Recursive chunking | LangChain `RecursiveCharacterTextSplitter` | ◐ `chunking/` (no generic recursive splitter yet) |
| Semantic chunking | LlamaIndex `SemanticSplitterNodeParser` | ✗ slot in `chunking/` |
| Late chunking | Jina embeddings | ✗ couples `chunking/` + `embeddings/` |
| Hybrid dense+BM25 | `rank_bm25`, `EnsembleRetriever` | ◐ `rag/retriever.py` has `use_keyword_search` flag (unimplemented) |
| RRF fusion | ~10 LOC | ✗ belongs in `rag/retriever.py` |
| Cross-encoder rerank | `sentence-transformers` | ✅ `rag/re_ranker.py` |
| ColBERT | ColBERT, RAGatouille | ✗ extend `re_ranker.py` |
| monoT5 | `pygaggle` | ✗ `re_ranker.py` |
| BGE/Cohere rerank | `FlagEmbedding`, Cohere | ✅ `re_ranker.py` (model swap) |
| HyDE | LangChain | ✗ near `prompt_builder.py`/retriever |
| Multi-query | LangChain | ✗ retriever/prompt layer |
| Self-RAG / CRAG / RAG-Fusion | LangGraph templates | ✗ orchestration-level |
| GraphRAG | `microsoft/graphrag` | ✗ new subsystem |
| RAGAS / ARES | `ragas`, `ARES` | ✅ `rag/evaluation.py` |
| Embedding models | `sentence-transformers`, `FlagEmbedding`, OpenAI | ✅ `embeddings/{local,openai}_embeddings.py` |
| Vector stores FAISS/Chroma | `faiss`, `chromadb` | ✅ `rag/storage/{faiss_store,chroma_store}.py` (implemented this session) |

Legend: ✅ implemented · ◐ partial/stub · ✗ not present (candidate work).

## 10. Recommended next steps for this repo (research-driven)

1. **Hybrid retrieval + RRF** — `use_keyword_search` is a dead flag; wire `rank_bm25`
   and fuse with RRF. Highest ROI, small surface (`rag/retriever.py`).
2. **Real cross-encoder reranker** — confirm `rag/re_ranker.py` uses a real
   `sentence-transformers` CrossEncoder (e.g. `ms-marco-MiniLM`) vs a stub.
3. **RAGAS evaluation** — back `rag/evaluation.py` with `ragas` for faithfulness /
   context-precision rather than ad-hoc metrics.
4. **Recursive + semantic chunking** — add `RecursiveCharacterTextSplitter`-style
   and embedding-distance semantic splitters to `chunking/`.
5. **HyDE / multi-query** — cheap retrieval-quality wins at the query layer.

## Caveats (verify before formal citation)
- *Query Rewriting* (2305.14283): arXiv ID from prior context, not re-fetched this session — verify.
- *RAG-Fusion* (2402.03367): short/independent arXiv paper, primarily a blog-popularized pattern — present as such.
- Recursive/semantic chunking, multi-query, hybrid search: **no canonical academic paper** — attribute to libraries.
- Cohere Rerank, OpenAI text-embedding-3: closed/hosted, no paper — official pages only.
- Repo "slot" judgments for `retriever.py`/`re_ranker.py` are based partly on filenames + flags; open the files to confirm stub vs. real before relying on them.
