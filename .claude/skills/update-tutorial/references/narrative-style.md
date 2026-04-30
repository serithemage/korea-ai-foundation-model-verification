# Narrative Style Guide for `docs/tutorial/`

The tutorial is a chronological learning log. Future readers should be able to *read it as a story*, not consult it as a report. Bullets and tables are supporting tools, never the primary form.

## The Rule

Walk the reader through the discovery: what was the question, what did we look at, what did we find, what does it mean, what doesn't fit yet. Cause and effect. Comparison and contrast. The narrator's path of reasoning.

## Anti-Pattern (do not do this)

```markdown
### 분석 결과

| 모델 | vocab_size | 결과 |
|------|-----------|------|
| Solar | 196,608 | From scratch |
| Mixtral | 32,000 | 비교 대상 |

- Vocabulary 크기가 다름
- Special token 패턴이 다름
- BPE merge rules가 다름
```

This is a report. Three bullets and a table tell the reader nothing about *why* it matters or how the conclusion was reached.

## Target Style (do this)

```markdown
### 분석 결과

Solar-Open-100B의 tokenizer를 분석해보니 vocabulary 크기가 196,608개로 확인되었습니다. 비교 대상 Mixtral의 32,000개와 비교하면 6배 이상 큰 규모입니다.

이렇게 큰 차이가 나는 이유를 생각해보면, Solar가 한국어를 포함한 다국어 처리를 위해 독자적으로 tokenizer를 설계했기 때문으로 보입니다. 만약 기존 모델의 tokenizer를 재사용했다면 vocabulary 크기가 이렇게 다를 수 없습니다.

Special token도 살펴보면 Solar는 `<s>`, `</s>`, `<pad>` 형식을 사용하는데, 이는 SentencePiece의 전통적인 방식입니다. Llama 3가 `<|begin_of_text|>` 같은 새로운 형식을 도입한 것과 대비됩니다.
```

The same facts — but now they form an argument.

## Authoring Tips

1. **Cause and effect.** Replace "X is Y" with "X is Y because Z" or "X is Y, which means Z". Every claim earns the next one.
2. **Comparison and contrast.** "X와 달리 Y는...", "Mixtral의 N과 비교하면..." — differences carry the weight of the conclusion.
3. **Show the path of reasoning.** "처음에는 A라고 봤지만, 살펴보니 B였다" beats "결과: B". Readers retain the reasoning, not the verdict.
4. **Address the reader directly.** "살펴보면", "확인해보니", "흥미로운 점은" — small signals that this is a guided tour, not a spec sheet.
5. **Tables for *summary numbers only*.** If a 3-row table can be replaced by one sentence ("vocab_size는 Solar 196,608 vs Mixtral 32,000으로 6배 차이"), do that instead. Reserve tables for cases where the reader genuinely needs to scan multiple rows side-by-side.
6. **No bullet lists for findings.** Bullets fragment the reasoning. If you have three findings, write three connected paragraphs.

## When Tables Are Acceptable

- Comparing 4+ models across 3+ dimensions where the reader will scan
- Listing config values (e.g., `num_layers: 64, hidden_size: 5120`)
- Final summary at the end of a long Q&A — *after* the narrative has been told

## Quality Check

After writing, ask: **"Could a reader who skipped this Q&A entirely later catch up by reading just my prose?"** If the answer requires also reading the bullets, rewrite the bullets into prose.
