# Update Tutorial Skill

**Purpose**: Q&A 형식의 검증 튜토리얼을 주제별 파일에 추가

## Description

이 skill은 검증 과정에서 진행된 Q&A를 `docs/tutorial/` 폴더의 주제별 파일에 추가합니다.

## Instructions

사용자가 `/update-tutorial` 명령을 실행하면:

1. 현재 대화에서 진행된 주요 Q&A를 식별합니다
2. Q&A의 주제를 분류합니다
3. 해당 주제의 파일에서 `<!-- SECTION_MARKER -->` 위치를 찾습니다
4. 새로운 Q&A를 마커 바로 위에 추가합니다
5. `docs/tutorial/README.md`의 목차를 업데이트합니다

## 튜토리얼 파일 구조

```
docs/tutorial/
├── README.md                      # 인덱스 + 목차
├── 01-기초개념.md                  # Q0, Q1 등
├── 02-tokenizer-분석.md           # Q2, Q5, Q6, Q10 등
├── 03-weight-architecture-분석.md # Q3, Q4, Q7, Q12 등
├── 04-사례연구.md                  # Q8, Q9 등
└── 05-방법론-평가.md               # Q11 등
```

## 주제 분류 기준

| 주제 키워드 | 대상 파일 |
|------------|----------|
| from scratch 개념, 검증 방법 개요, 논란 배경 | 01-기초개념.md |
| Tokenizer, vocabulary, vocab_size, merge rules, BPE, special tokens | 02-tokenizer-분석.md |
| Weight, cosine similarity, architecture, MoE, layer, expert | 03-weight-architecture-분석.md |
| 특정 모델명 (Solar, HyperCLOVAX, A.X-K1 등), 논란 사례, 검증 결과 | 04-사례연구.md |
| 방법론 비판, 학술 연구, 개선 방향, 한계점 | 05-방법론-평가.md |

## 작성 스타일 가이드

### 내러티브(서술형) 형식 원칙

**반드시 서술형으로 작성합니다.** 열거형(bullet point, 테이블 나열)보다 **이야기를 풀어가듯 설명**하는 형식을 사용합니다.

#### 피해야 할 스타일 (열거형)

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

#### 사용해야 할 스타일 (내러티브)

```markdown
### 분석 결과

Solar-Open-100B의 tokenizer를 분석해보니, vocabulary 크기가 196,608개로 확인되었습니다. 이는 비교 대상인 Mixtral의 32,000개와 비교하면 6배 이상 큰 규모입니다.

이렇게 큰 차이가 나는 이유를 생각해보면, Solar가 한국어를 포함한 다국어 처리를 위해 독자적으로 tokenizer를 설계했기 때문으로 보입니다. 만약 기존 모델의 tokenizer를 재사용했다면, vocabulary 크기가 이렇게 다를 수 없습니다.

Special token도 살펴보면, Solar는 `<s>`, `</s>`, `<pad>` 형식을 사용하는데, 이는 SentencePiece의 전통적인 방식입니다. Llama 3가 `<|begin_of_text|>` 같은 새로운 형식을 도입한 것과 대비됩니다.
```

### 내러티브 작성 팁

1. **인과관계를 설명합니다**: "A이다" 대신 "A인데, 그 이유는 B 때문이다"
2. **비교와 대조를 활용합니다**: "X와 달리 Y는..." 형식으로 차이점을 부각
3. **발견 과정을 서술합니다**: "처음에는 A라고 생각했지만, 분석해보니 B였다"
4. **독자에게 말하듯 씁니다**: "살펴보면", "확인해보니", "흥미로운 점은"
5. **테이블은 보조 수단으로만**: 핵심 수치 요약에만 사용하고, 주 설명은 서술형으로

## Q&A Format

```markdown
---

## Q{N}: {질문 제목}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{내러티브 형식의 답변 - 이야기를 풀어가듯 서술}

---
```

## Process

1. `docs/tutorial/README.md`를 읽어 현재 Q&A 개수를 파악합니다
2. 대화에서 검증 관련 질문을 식별합니다
3. 질문의 주제를 분류하여 대상 파일을 결정합니다
4. 해당 파일의 `<!-- SECTION_MARKER -->` 위에 새 Q&A를 **내러티브 형식으로** 삽입합니다
5. `docs/tutorial/README.md`의 목차에 새 Q&A 항목을 추가합니다

## Example Usage

```
User: /update-tutorial
```

이 명령 실행 시, 현재 세션에서 진행된 검증 관련 Q&A를 적절한 주제 파일에 내러티브 형식으로 추가합니다.

## 새 주제 파일 생성

기존 분류에 맞지 않는 새로운 주제가 나타나면:

1. `docs/tutorial/0N-새주제.md` 파일을 생성합니다
2. 파일 하단에 `<!-- SECTION_MARKER -->` 마커를 추가합니다
3. `docs/tutorial/README.md` 목차에 새 섹션을 추가합니다
4. 이 스킬 파일의 "주제 분류 기준" 테이블을 업데이트합니다
