# Prompt Engineering Guide — the six-W framework

A practical, professional method for writing prompts that behave predictably.
The core idea: **interrogate your prompt with six questions before you send it.**
If any answer is missing or vague, the model will guess — and its guess is your
bug.

---

## 1. The six W-questions

For every prompt — whether you are instructing an AI agent to do a task, or
calling an LLM inside a product — make sure the prompt answers all six:

| Question | What it pins down | Symptom when missing |
|---|---|---|
| **What** | The exact deliverable: the artifact, its shape, its scope. | Model returns the wrong *kind* of thing (prose when you wanted a table). |
| **Why** | The success criterion / intent behind the task. | Technically-correct-but-useless answers; model optimises the wrong thing. |
| **Who** | The role/persona and the audience. | Wrong altitude — too junior, too academic, wrong tone. |
| **When** | Ordering, triggers, and temporal scope ("current", "as of 2025"). | Stale facts; steps done out of order. |
| **Where** | The context and constraints: inputs available, environment, boundaries. | Model invents context, or strays outside the intended surface. |
| **How** | The method, the output format, and the limits (length, schema, what NOT to do). | Unparseable output; scope creep; hallucinated structure. |

A useful mnemonic: **What + Why** define *success*, **Who + Where** define
*context*, **When + How** define *execution*.

## 2. Anatomy of a strong prompt

A reliable prompt usually has these blocks, in roughly this order:

1. **Role** (*Who*) — "You are a read-only code-audit agent."
2. **Objective** (*What* + *Why*) — "Find modules that fail to import, so we can
   tell real bugs from missing dependencies."
3. **Context / inputs** (*Where* + *When*) — paths, versions, "use the venv at
   …", "best practices as of 2026".
4. **Method** (*How*) — the steps or strategy to follow.
5. **Output contract** (*How*) — exact format, schema, length; what to include
   and exclude.
6. **Guardrails** (*How*, negative space) — "Do not fix anything." "If unsure,
   say so — do not fabricate."

## 3. Worked transformation: weak → strong

**Weak (under-specified):**

> Find problems in the RAG repo.

Run the six W's against it: *What* problems? *Why* (what will we do with them)?
*Who* are you? *Where* is the repo / how to run it? *How* should the output
look? Almost everything is missing — so the model improvises.

**Strong:**

> *(Who)* You are a read-only code-audit agent.
> *(What/Why)* Identify every Python module under `pipeline/src` that fails to
> import, and classify each failure as a missing third-party dependency versus a
> real code bug — we need to know which are environment issues and which we must
> fix.
> *(Where/When)* Use the project venv at `/…/.venv/bin/python` with
> `PYTHONPATH=pipeline/src`; the repo is at `/…/RAG`.
> *(How)* Convert each file path to a dotted module and attempt
> `importlib.import_module`. Capture the exception type and the root-cause line.
> *(How — output contract)* Return a table of `module → failure class → one-line
> cause`, plus a list of the distinct missing pip packages.
> *(Guardrail)* Do not fix anything; report only. If a cause is ambiguous, say so.

This is essentially the real audit prompt used in this project — see
[`agent_prompts.md`](agent_prompts.md).

## 4. Common failure modes (and the W that fixes them)

| Failure mode | Missing W | Fix |
|---|---|---|
| Output can't be parsed / inconsistent shape | **How** | Specify an exact format or JSON schema; give a tiny example. |
| Confidently wrong / invented facts or APIs | **Where / How** | Provide the real context; add "if unsure, say so; do not fabricate." |
| Right content, wrong depth or tone | **Who** | State the role and the audience explicitly. |
| Model "improves" beyond what you asked | **What / How** | Bound the scope; list non-goals ("do not refactor"). |
| Stale or time-ambiguous answer | **When** | Anchor the timeframe ("current", "as of <date>"). |
| Verbose, unfocused answer | **Why / How** | State the intent and a length/format limit. |

## 5. Techniques that complement the framework

These map onto the W's; reach for them when a single instruction is not enough:

- **Few-shot examples** (sharpen *How* and *What*): show 1–3 input→output pairs.
- **Chain-of-thought / "think step by step"** (*How*): for multi-step reasoning;
  but request the *final* answer in a fixed format so it stays parseable.
- **Structured output / schemas** (*How*): force JSON to remove format ambiguity.
- **Role assignment** (*Who*): "You are a security reviewer" changes priorities.
- **Decomposition** (*What*): split a big task into smaller, individually
  checkable prompts (this is exactly how the repo audit was fanned out to agents).
- **Self-critique / verification pass** (*Why*): ask the model to check its own
  output against the success criterion before finalising.
- **Negative instructions** (*How*): say what *not* to do; models honour explicit
  boundaries better than implied ones.

## 6. The pre-flight checklist

Before sending any prompt, confirm you can answer:

- [ ] **What** exact artifact should come back, and at what scope?
- [ ] **Why** does it matter — what is the one-line success criterion?
- [ ] **Who** is the model acting as, and who is the audience?
- [ ] **When** — any ordering or timeframe to anchor?
- [ ] **Where** — is every input/context/constraint the model needs actually in
      the prompt (not in your head)?
- [ ] **How** — is the output format/length specified, and are the non-goals and
      guardrails explicit?

If two or more boxes are blank, the prompt is not ready.

## 7. Further reading

- Anthropic — Prompt engineering overview:
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview
- OpenAI — Prompt engineering guide:
  https://platform.openai.com/docs/guides/prompt-engineering
- *Chain-of-Thought Prompting Elicits Reasoning in LLMs*, Wei et al., 2022 —
  https://arxiv.org/abs/2201.11903
