# Update Tutorial Skill

**Purpose**: Q&A 형식의 검증 튜토리얼을 docs/00-tutorial.md에 추가

## Description

이 skill은 Solar-Open-100B 검증 과정에서 진행된 Q&A를 `docs/00-tutorial.md` 튜토리얼 문서에 추가합니다.

## Instructions

사용자가 `/update-tutorial` 명령을 실행하면:

1. 현재 대화에서 진행된 주요 Q&A를 식별합니다
2. `docs/00-tutorial.md`의 `<!-- TUTORIAL_MARKER -->` 위치를 찾습니다
3. 새로운 Q&A를 마커 바로 위에 추가합니다
4. Q&A 번호를 자동으로 증가시킵니다

## Target File

**튜토리얼 파일**: `docs/00-tutorial.md`

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

### Q&A Format

```markdown
---

## Q{N}: {질문 제목}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{내러티브 형식의 답변 - 이야기를 풀어가듯 서술}

---
```

## Process

1. `docs/00-tutorial.md` 파일을 읽어서 현재 Q&A 개수를 파악합니다
2. 대화에서 사용자가 물어본 검증 관련 질문을 식별합니다
3. 해당 질문과 답변을 **내러티브(서술형) 형식**으로 정리합니다
4. `<!-- TUTORIAL_MARKER -->` 라인 바로 위에 새 Q&A를 삽입합니다
5. `docs/00-tutorial.md`를 업데이트합니다

## Example Usage

```
User: /update-tutorial
```

이 명령 실행 시, 현재 세션에서 진행된 검증 관련 Q&A를 `docs/00-tutorial.md`에 내러티브 형식으로 추가합니다.
