---
name: update-changelog
description: Use when the user invokes /update-changelog, or when a session has produced concrete repository changes (file edits, verification results, methodology updates, structural changes) that should be recorded in CHANGELOG.md by date and category.
---

# update-changelog

Append the session's substantive changes to `CHANGELOG.md`. Newest date first; entries grouped by category under each date.

## When to Use

- User runs `/update-changelog`
- A session ended with real repo changes worth a permanent log entry: a verification was completed, a methodology section was added, a tutorial Q&A landed, a new skill or directory appeared

Skip for: pure conversation, trivial typo fixes, untracked exploration.

## Workflow

1. Read `CHANGELOG.md` to see the current structure and the most recent date.
2. Identify substantive changes from the session: file additions/edits, verification outcomes, doc updates, structural moves.
3. Group by category (table below).
4. Today's date already present? Append under it. Otherwise create a new dated section directly under the top `---` divider.
5. Write each entry as one concise line; include a markdown link if there's an external reference (paper, GitHub issue).

## Format

```markdown
---

## {YYYY-MM-DD}

### {Category}
- {Change 1}
- {Change 2 with [link](https://...) if relevant}

### {Other Category}
- {Change}

---
```

## Categories

| Category | Use for |
|---|---|
| **모델 검증** | Specific-model verification work (e.g., "Solar-Open-100B 검증 완료") |
| **문서 추가/수정** | Tutorial, README, wiki content (e.g., "튜토리얼 Q11 추가") |
| **방법론 개선** | Methodology changes, new analysis approach |
| **프로젝트 구조** | File/folder reorganization, skill/command additions |
| **버그 수정** | Errors corrected (links, typos that materially mislead) |

## Authoring Rules

1. **One concise line per entry** — the *what*, not the *how*
2. **Be specific** — "Q11 방법론 비판 섹션 추가" beats "문서 수정"
3. **Link external references** — papers, arXiv IDs, GitHub issues belong inline
4. **No duplicates** — if it's already in the changelog, don't re-add

## Example

```markdown
## 2026-01-05

### 방법론 개선
- 검증 방법론 한계와 학술 연구 결과 섹션 추가
- [arXiv:2502.00706](https://arxiv.org/abs/2502.00706) Black-box Output Similarity Testing 참조
- Yi-Llama 논쟁 사례 연구 추가

### 문서 추가
- 튜토리얼 Q11: 방법론 비판에 대한 답변 추가

### 프로젝트 구조
- `.claude/skills/update-changelog/` skill 추가
```

## Common Mistakes

- Creating a new dated section when today's already exists — append instead
- Logging non-changes (questions answered without repo changes) — skip them
- Vague entries ("문서 수정") — name the section or Q-number
- Forgetting the trailing `---` divider — breaks future appends
- Inserting today's date below an older date — newest must stay on top
