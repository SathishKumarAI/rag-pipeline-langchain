# Prompts

A curated, teaching-oriented collection of the prompts used to build and operate
this RAG project — written in plain professional English so they can be reused,
critiqued, and improved.

Every prompt here is analysed with the **six W-questions** — *What, Why, Who,
When, Where, How* — because a prompt that cannot answer those six is usually
under-specified.

## Files

| File | What it gives you |
|---|---|
| [`prompt_engineering_guide.md`](prompt_engineering_guide.md) | The framework: the six W-questions, an anatomy of a strong prompt, common failure modes, and a reusable checklist. **Read this first.** |
| [`agent_prompts.md`](agent_prompts.md) | The real subagent prompts used during this build (import audit, web research), each annotated with its six W's and a concrete "how to improve" rewrite. |
| [`rag_system_prompts.md`](rag_system_prompts.md) | Production prompt templates for the RAG pipeline itself (answer generation, query rewriting, grading, faithfulness eval), each with W-form analysis and improvements. |

## How to use this collection

1. Read the **guide** to internalise the six-W framework.
2. Study the **agent prompts** to see the framework applied to *task* prompts
   (instructing an AI worker).
3. Study the **system prompts** to see it applied to *product* prompts (the LLM
   calls inside the RAG product).
4. When you write a new prompt, run it through the checklist at the end of the
   guide before using it.

## The one-line rule

> A good prompt tells the model **what** to produce, **why** it matters (the
> success criterion), **who** it is acting as, **when/where** the output will be
> used (context + constraints), and **how** to format and bound the answer.
