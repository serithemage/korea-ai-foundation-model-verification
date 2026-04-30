---
name: update-tutorial
description: Use when the user invokes /update-tutorial, or when a verification-related Q&A exchange has just completed in the session and should be persisted into docs/tutorial/ as narrative-style content. Classifies by topic, finds the right file, inserts above the SECTION_MARKER, and updates the README index.
---

# update-tutorial

Append the session's Q&A to `docs/tutorial/` in narrative prose (not bullet/table dumps). The tutorial is a chronological learning log; future readers should be able to read it as a story, not a report.

## When to Use

- User runs `/update-tutorial`
- A substantive verification-related Q&A just happened — concept question, methodology, model-specific finding, methodology critique
- Clarifying a term that future sessions will likely reference

Skip for project-operational questions ("어떤 파일이 어디 있어?"), Claude usage questions, or yes/no confirmations.

## Workflow

1. Identify the session's Q&A pairs that belong in the tutorial.
2. Classify each by topic (table below) → choose target file.
3. Read `docs/tutorial/README.md` to find the next Q-number (highest existing + 1, shared across files).
4. Open the target file, locate `<!-- SECTION_MARKER -->`, insert the new Q&A **above** the marker.
5. Append a one-line entry to the index in `docs/tutorial/README.md`.

## Topic Classification

| Topic keywords | Target file |
|---|---|
| from-scratch concept, verification overview, controversy background | `01-기초개념.md` |
| Tokenizer, vocabulary, vocab_size, BPE, merge rules, special tokens | `02-tokenizer-분석.md` |
| Weight, cosine similarity, architecture, MoE, layer, expert | `03-weight-architecture-분석.md` |
| Specific model verification result, Solar/HyperCLOVAX/A.X-K1 cases | `04-사례연구.md` |
| Methodology critique, academic research, improvement direction | `05-방법론-평가.md` |

If none fit: create `docs/tutorial/0N-새주제.md` with a `<!-- SECTION_MARKER -->` line at the bottom, add the README section, and update this skill's table.

## Q&A Format

```markdown
---

## Q{N}: {질문 요약}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{narrative prose — see references/narrative-style.md}

---
```

## Narrative Style

**Required**: write answers as connected prose that walks the reader through the discovery. Bullets and tables are a supporting tool, not the primary form.

See `references/narrative-style.md` for full examples and authoring tips.

## Common Mistakes

- Bullet/table dump instead of narrative — defeats the tutorial's purpose
- Inserting Q&A below the `SECTION_MARKER` instead of above — breaks future appends
- Reusing a Q-number — Q-numbers are global across all five files
- Forgetting to update `docs/tutorial/README.md` index — orphan entry
- Adding tutorial entries for ephemeral project-operational chat (not real learning)
