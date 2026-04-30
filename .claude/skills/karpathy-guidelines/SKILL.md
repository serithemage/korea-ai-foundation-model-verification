---
name: karpathy-guidelines
description: Use when writing, reviewing, or refactoring code in this repo to reduce common LLM coding mistakes — overcomplication, drive-by edits to unrelated lines, hidden assumptions, vague success criteria. Four principles: Think Before Coding, Simplicity First, Surgical Changes, Goal-Driven Execution.
license: MIT
source: https://github.com/forrestchang/andrej-karpathy-skills
---

# Karpathy Guidelines

Behavioral guidelines to reduce common LLM coding mistakes, derived from [Andrej Karpathy's observations](https://x.com/karpathy/status/2015883857489522876) on LLM coding pitfalls. Adapted from [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills) (MIT).

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks (typo fix, obvious one-liner), use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Working in This Repo

This project is verification documentation + LLM-DNA experiments — most "code" is markdown, YAML, or short Python scripts. The principles still apply:

- **Think Before Coding**: When the user reports an unexpected result (e.g., the Solar/Mixtral identical-DNA issue), surface the diagnosis and tradeoffs (re-extract vs. exclude vs. switch dataset) before acting.
- **Simplicity First**: For one-shot scripts in `experiments/llm-dna/scripts/`, prefer 50 readable lines over 200 lines of "configurable" abstraction.
- **Surgical Changes**: Don't reformat or rewrite `wiki/` pages while editing one; don't restructure tutorial files while adding one Q&A.
- **Goal-Driven Execution**: SageMaker spot jobs need explicit success criteria up-front (e.g., "DNA extraction produces non-empty `signature` of length 128 with non-trivial variance"); don't claim success on `Status=Completed` alone.

## Working / Not Working

| Working | Not working |
|---|---|
| Diff contains only what the user asked for | Drive-by reformatting in unrelated files |
| Clarifying question came before edits | Discovered wrong assumption after the work landed |
| 50-line script that solves the task | 200-line script with unused configuration knobs |
| Verifiable success criteria stated up front | "Make it work" with no defined check |
