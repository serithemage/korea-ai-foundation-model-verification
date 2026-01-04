# Solar-Open-100B 검증 프로젝트 - Claude 지침

## 프로젝트 목적

이 프로젝트는 Upstage의 Solar-Open-100B 모델이 실제로 "from scratch"로 학습되었는지 검증합니다.
검증 과정의 모든 Q&A는 튜토리얼 형태로 README.md에 기록됩니다.

## 질문 처리 워크플로우

### 사용자 입력 유형 판별

```
사용자 입력
    │
    ├─ 질문인가? (?, ~인가요, ~뭔가요, 설명해줘 등)
    │   │
    │   └─ YES → Perplexity MCP로 조사 → README 튜토리얼에 Q&A 추가
    │
    └─ 명령인가? (실행해, 분석해, 만들어 등)
        │
        └─ YES → 명령 실행 (튜토리얼 업데이트 없음)
```

### Perplexity MCP 조사 규칙

**중요**: 검증 관련 질문을 받으면 반드시 Perplexity MCP를 사용하여 조사합니다.

1. **영문으로 쿼리 작성** (사용자의 전역 CLAUDE.md 지침)
2. **구체적이고 기술적인 질문으로 변환**
3. **조사 결과를 한국어로 정리하여 README에 추가**

예시:
- 사용자: "Tokenizer 분석에 대해서 자세하게 설명해줘"
- Perplexity 쿼리: "How to analyze LLM tokenizers to determine if a model was trained from scratch or fine-tuned, including BPE, SentencePiece comparison techniques"

### 튜토리얼에 기록해야 하는 Q&A 유형

다음 주제에 대한 질문과 답변은 반드시 README.md 튜토리얼에 추가:

1. **LLM 학습 검증 방법론** - from scratch vs fine-tuning 판별법
2. **Tokenizer 분석** - vocabulary 비교, BPE merge 분석
3. **Weight 분석** - cosine similarity, 해싱, PCA
4. **Architecture 분석** - config 비교, 구조적 특징
5. **Solar-Open-100B 구체적 분석 결과** - 실제 검증 수행 결과
6. **MoE 모델 관련 지식** - Mixture-of-Experts 아키텍처

### Q&A 추가 프로세스

1. **Perplexity MCP로 조사** (영문 쿼리)
2. **조사 결과를 구조화된 한국어 답변으로 정리**
3. README.md의 `<!-- TUTORIAL_MARKER -->` 위에 새 Q&A 추가
4. Q&A 번호는 기존 Q&A 개수 + 1
5. 형식:

```markdown
---

### Q{N}: {질문 요약}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{Perplexity MCP 조사 기반 구조화된 답변}

---
```

### 검증 진행 상황 업데이트

검증 작업 수행 시 README.md의 "검증 진행 상황" 체크리스트도 함께 업데이트

## 주요 파일

- `README.md` - 프로젝트 문서 및 튜토리얼
- `.claude/skills/update-tutorial.md` - 튜토리얼 업데이트 skill
- `.claude/settings.json` - hooks 설정

## 검증 대상 모델

- **모델**: upstage/Solar-Open-100B
- **URL**: https://huggingface.co/upstage/Solar-Open-100B
- **공식 주장**: "Trained Entirely from Scratch"
- **학습 토큰**: 19.7T tokens
- **아키텍처**: MoE (102.6B total, 12B active)
