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
    │   └─ YES → Perplexity MCP 조사 (영문) → docs/00-tutorial.md에 Q&A 추가
    │
    └─ 명령인가? (실행해, 분석해 등)
        └─ YES → 명령 실행 (튜토리얼 업데이트 없음)
```

### Perplexity MCP 규칙

1. **영문으로 쿼리 작성** (글로벌 CLAUDE.md 지침)
2. **조사 결과를 한국어로 정리**
3. `docs/00-tutorial.md`의 `<!-- TUTORIAL_MARKER -->` 위에 Q&A 추가

### Q&A 형식

```markdown
---

## Q{N}: {질문 요약}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{구조화된 답변}

---
```

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
| `docs/00-tutorial.md` | Q&A 튜토리얼 (자동 업데이트 대상) |
| `docs/01-tokenizer-analysis.md` | Tokenizer 분석 방법론 + 모델별 결과 |
| `docs/02-weight-analysis.md` | Weight 분석 방법론 + 모델별 결과 |
| `docs/03-architecture-analysis.md` | Architecture 분석 방법론 + 모델별 결과 |
| `docs/04-behavior-analysis.md` | 행동 분석 방법론 + 모델별 결과 |
| `.claude/skills/update-tutorial.md` | 튜토리얼 업데이트 skill |
| `.claude/commands/*.md` | 커스텀 커맨드 정의 |

## 검증 작업 시 참고

- 새 모델 검증 시: 각 분석 문서(01-04)의 "모델별 검증 결과" 섹션에 추가
- 분석 완료 후: README.md의 검증 대상 모델 테이블 상태 업데이트
- Q&A 발생 시: docs/00-tutorial.md에 자동 기록
