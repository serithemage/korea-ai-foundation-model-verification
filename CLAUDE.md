# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**국가 AI 파운데이션 모델 "From Scratch" 검증 프로젝트**

한국 정부의 국가 AI 파운데이션 모델 프로젝트에 참여한 5개 기관의 공개 모델이 실제로 "from scratch"로 학습되었는지 검증합니다.

| 모델 | 상태 | 판정 |
|------|------|------|
| Upstage Solar-Open-100B | ✅ 완료 | From scratch 신뢰 |
| NAVER HyperCLOVAX-SEED-Think-32B | ⚠️ 진행중 | 부분적 재사용 |
| SKT A.X-K1 | 📋 대기 | - |
| NC AI VAETKI | 📋 대기 | - |
| LG AI K-EXAONE | 📋 대기 | - |

## 사용 가능한 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/commit-push` | 변경사항 분석 후 커밋 & 푸시 |
| `/save` 또는 `/save {메시지}` | 빠른 커밋 & 푸시 |
| `/update-tutorial` | Q&A 튜토리얼 수동 업데이트 |

## 질문 처리 워크플로우

```
사용자 입력
    │
    ├─ 질문인가? (?, 설명해줘, ~인가요 등)
    │   └─ YES → Perplexity MCP 조사 (영문) → docs/tutorial/에 Q&A 추가
    │
    └─ 명령인가? (실행해, 분석해 등)
        └─ YES → 명령 실행 (튜토리얼 업데이트 없음)
```

### Perplexity MCP 규칙

1. **영문으로 쿼리 작성** (글로벌 CLAUDE.md 지침)
2. **조사 결과를 한국어로 정리**
3. 주제에 맞는 `docs/tutorial/*.md` 파일의 `<!-- SECTION_MARKER -->` 위에 Q&A 추가

### Q&A 주제별 분류

| 주제 | 대상 파일 |
|------|----------|
| from scratch 개념, 검증 방법 개요 | `docs/tutorial/01-기초개념.md` |
| Tokenizer, vocabulary, merge rules | `docs/tutorial/02-tokenizer-분석.md` |
| Weight, cosine similarity, architecture | `docs/tutorial/03-weight-architecture-분석.md` |
| 특정 모델 검증 결과, 논란 사례 | `docs/tutorial/04-사례연구.md` |
| 방법론 비판, 학술 연구, 개선 방향 | `docs/tutorial/05-방법론-평가.md` |

### Q&A 형식 (내러티브 스타일)

```markdown
---

## Q{N}: {질문 요약}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{내러티브 형식 - 이야기를 풀어가듯 서술}

---
```

**작성 원칙**: 테이블/bullet point 나열보다 **서술형**으로 작성. 인과관계 설명, 비교/대조 활용.

## 검증 방법론

4가지 분석 방법으로 from scratch 여부를 판별:

1. **Tokenizer 분석** (docs/01-tokenizer-analysis.md)
   - Vocabulary 비교, BPE merge rules, special tokens 패턴

2. **Weight 분석** (docs/02-weight-analysis.md)
   - Cosine similarity, tensor 해시 비교, PCA 분포

3. **Architecture 분석** (docs/03-architecture-analysis.md)
   - config.json 비교, MoE 구조, RoPE/Attention 설정

4. **행동 분석** (docs/04-behavior-analysis.md)
   - Knowledge cutoff, refusal pattern, safety alignment

각 분석 문서에는 **"모델별 검증 결과"** 섹션이 있어 5개 모델의 분석 결과를 기록합니다.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `docs/tutorial/README.md` | Q&A 튜토리얼 인덱스 |
| `docs/tutorial/01-기초개념.md` | 기초 개념 Q&A |
| `docs/tutorial/02-tokenizer-분석.md` | Tokenizer 분석 Q&A |
| `docs/tutorial/03-weight-architecture-분석.md` | Weight/Architecture Q&A |
| `docs/tutorial/04-사례연구.md` | 모델 검증 사례 Q&A |
| `docs/tutorial/05-방법론-평가.md` | 방법론 평가 Q&A |
| `docs/01-tokenizer-analysis.md` | Tokenizer 분석 방법론 + 모델별 결과 |
| `docs/02-weight-analysis.md` | Weight 분석 방법론 + 모델별 결과 |
| `docs/03-architecture-analysis.md` | Architecture 분석 방법론 + 모델별 결과 |
| `docs/04-behavior-analysis.md` | 행동 분석 방법론 + 모델별 결과 |
| `.claude/skills/update-tutorial.md` | 튜토리얼 업데이트 skill |
| `.claude/commands/*.md` | 커스텀 커맨드 정의 |

## 검증 작업 시 참고

- 새 모델 검증 시: 각 분석 문서(01-04)의 "모델별 검증 결과" 섹션에 추가
- 분석 완료 후: README.md의 검증 대상 모델 테이블 상태 업데이트
- Q&A 발생 시: `docs/tutorial/` 해당 주제 파일에 기록
