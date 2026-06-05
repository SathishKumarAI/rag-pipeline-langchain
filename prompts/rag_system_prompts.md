# RAG System Prompts — production templates, annotated

Reusable prompt templates for the LLM calls *inside* the RAG product, each
analysed with the six W-questions (see
[`prompt_engineering_guide.md`](prompt_engineering_guide.md)) and followed by
improvements. These complement the techniques in
[`../docs/research/rag_best_practices.md`](../docs/research/rag_best_practices.md).

> The repo currently builds a minimal prompt
> (`f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"` in
> `rag_pipeline/lab_pipeline.py`). The templates below are the professional
> upgrades, with the reasoning made explicit.

---

## 1. Answer generation (grounded, citation-required)

```text
You are a precise assistant answering strictly from the provided context.

Context:
{context}

Question: {question}

Instructions:
- Answer using ONLY the context above. Do not use outside knowledge.
- If the context does not contain the answer, reply exactly:
  "I don't have enough information to answer that."
- Cite the supporting source after each claim using its [source_id].
- Be concise: at most {max_sentences} sentences.

Answer:
```

**Six-W analysis**
- **What:** a grounded, cited answer. **Why:** factuality — the whole point of
  RAG is to suppress hallucination by binding the answer to retrieved evidence.
- **Who:** "precise assistant answering strictly from context" — sets priorities.
- **Where:** the retrieved `{context}` is the *only* permitted knowledge source.
- **How:** explicit grounding rule, an **exact** abstain string (so the app can
  detect "no answer" deterministically), a citation requirement, and a length
  bound.

**Improvements**
1. Number/label each context passage (`[1] …`) so `[source_id]` citations are
   unambiguous.
2. Add: "If passages conflict, say so and present both." Real corpora contradict
   themselves.
3. For chat, prepend prior turns but keep the grounding rule scoped to the
   current question.

---

## 2. Query rewriting (rewrite-retrieve-read)

```text
Rewrite the user's question into a standalone search query optimised for
retrieval. Resolve pronouns and references using the conversation, expand
critical acronyms, and keep it under 30 words. Output ONLY the rewritten query.

Conversation:
{history}

User question: {question}
Rewritten query:
```

**Six-W analysis**
- **What:** one standalone retrieval query. **Why:** raw chat turns ("what about
  the second one?") retrieve poorly; a self-contained query improves recall.
- **When:** uses conversation history to resolve time/reference.
- **How:** "Output ONLY the rewritten query" keeps it parseable; the 30-word cap
  prevents the model from answering instead of rewriting.

**Improvements**
1. Multi-query variant: "Produce 3 diverse rewrites, one per line" → retrieve all,
   fuse with RRF (see research §3/§5).
2. Add a no-op guard: "If the question is already standalone, return it
   unchanged."

---

## 3. Retrieval grading (Corrective-RAG style)

```text
You are grading whether a retrieved passage is relevant to a question.

Question: {question}
Passage: {passage}

Reply with a JSON object only: {"relevant": true|false, "reason": "<short>"}.
Judge relevance by whether the passage could contribute to answering the
question, not by whether it fully answers it.
```

**Six-W analysis**
- **What:** a binary relevance judgement with a reason. **Why:** to filter noisy
  retrievals before generation (the CRAG pattern — research §6), cutting
  distractor-induced hallucination.
- **Who:** a grader/evaluator role narrows behaviour.
- **How:** **JSON-only** output (machine-consumable), and an explicit relevance
  definition so the bar is consistent across passages.

**Improvements**
1. Use a 3-way label (`correct | ambiguous | incorrect`) to drive CRAG's
   web-search fallback on `incorrect`.
2. Enforce the schema with a structured-output/JSON-mode API call rather than
   trusting free text.

---

## 4. Faithfulness evaluation (RAGAS-style judge)

```text
You are evaluating whether an answer is faithful to its supporting context.

Context:
{context}

Answer:
{answer}

Break the answer into atomic factual claims. For each claim, decide whether it
is supported by the context. Then output JSON only:
{"claims": [{"claim": "...", "supported": true|false}],
 "faithfulness": <supported_claims / total_claims as a 0-1 float>}.
```

**Six-W analysis**
- **What:** a per-claim support breakdown + a faithfulness score. **Why:** to
  measure hallucination quantitatively in evaluation (research §7), feeding
  `rag/evaluation.py`.
- **Who:** an evaluator/judge — distinct from the answering model (use a separate
  call to avoid self-grading bias).
- **How:** claim decomposition makes the score interpretable and auditable;
  JSON-only output makes it aggregatable across a test set.

**Improvements**
1. Run the judge with a *different* model than the generator to reduce
   correlated errors.
2. Add `answer_relevancy` and `context_precision` judges (RAGAS's other axes) for
   a fuller picture.
3. Calibrate against a small human-labelled set before trusting the numbers.

---

## Cross-cutting principles for product prompts

1. **Ground and give an escape hatch.** Always permit "I don't know" with an
   exact string the app can detect — a confident wrong answer is worse than an
   abstention.
2. **Make outputs machine-readable** (JSON/fixed format) wherever another step
   consumes them; enforce with the provider's structured-output mode.
3. **Separate roles across calls** — generator, grader, and judge should be
   distinct prompts (and ideally distinct models for evaluation).
4. **Bound length and scope** to control cost and prevent the model from
   wandering outside the retrieved evidence.
5. **Template the variables** (`{context}`, `{question}`, `{history}`) so prompts
   are versioned, testable artifacts — not strings buried in code.
