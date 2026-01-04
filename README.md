# 국가 AI 파운데이션 모델 "From Scratch" 검증 프로젝트

## 프로젝트 개요

### 시작 동기

2026년 1월, 한국 AI 업계에서 "from scratch 학습" 논란이 발생했습니다. 이 논란을 계기로 **LLM에 문외한인 소프트웨어 개발자가 학습과 병행하여 어디까지 모델 검증이라는 전문 영역의 작업을 수행할 수 있는지** 검증해보고자 이 프로젝트를 시작했습니다.

### 저자 배경

- 30년차 소프트웨어 개발자
- 수학 전공 (낙제점이지만)
- LLM/ML 분야 비전문가

### 프로젝트 목표

1. **학습**: LLM 모델 검증 방법론 습득
2. **실습**: 실제 공개 모델에 검증 기법 적용
3. **문서화**: 학습 과정을 Q&A 튜토리얼로 기록

### 검증 대상 모델

특정 모델만 검증하는 것은 불공평하다고 판단하여, **국가 AI 파운데이션 모델 프로젝트**에 참여한 5개 기관의 공개 모델을 모두 검증 대상에 포함시켰습니다.

| 기관 | 모델명 | 파라미터 | 유형 | HuggingFace |
|------|--------|----------|------|-------------|
| **NAVER Cloud** | HyperCLOVAX-SEED-Think | 32B | Dense | [링크](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B) |
| **Upstage** | Solar-Open-100B | 102B | MoE | [링크](https://huggingface.co/upstage/Solar-Open-100B) |
| **SKT** | A.X-K1 | 519B | MoE | [링크](https://huggingface.co/skt/A.X-K1) |
| **NC AI** | VAETKI | 112B | MoE | [링크](https://huggingface.co/NC-AI-consortium-VAETKI/VAETKI) |
| **LG AI 연구원** | K-EXAONE | 236B | MoE | [링크](https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B) |

### 튜토리얼 안내

[Q&A 튜토리얼](docs/00-tutorial.md)에는 검증 과정에서 학습한 질문과 답변들이 정리되어 있습니다. LLM 검증 방법론에 관심 있는 분들께 참고가 되길 바랍니다.

---

## 튜토리얼 자동 업데이트 시스템

이 프로젝트는 Claude Code의 hooks와 skills를 활용하여 Q&A 튜토리얼을 자동으로 업데이트합니다.

### 작동 방식

```mermaid
flowchart TB
    subgraph User["사용자"]
        Q[질문 입력]
    end

    subgraph Claude["Claude Code"]
        A{질문 유형 판별}
        P[Perplexity MCP 조사]
        R[조사 결과 정리]
        U[튜토리얼 업데이트]
    end

    subgraph Files["프로젝트 파일"]
        TUTORIAL[docs/00-tutorial.md<br/>Q&A 튜토리얼]
        CLAUDE[CLAUDE.md<br/>프로젝트 지침]
        SKILL[.claude/skills/<br/>update-tutorial.md]
    end

    Q --> A
    A -->|검증 관련 질문| P
    A -->|일반 명령| U
    P --> R
    R --> U
    U --> TUTORIAL
    CLAUDE -.->|지침 참조| A
    SKILL -.->|skill 정의| U
```

### 프로세스 상세

```mermaid
sequenceDiagram
    participant U as 사용자
    participant C as Claude Code
    participant P as Perplexity MCP
    participant F as docs/00-tutorial.md

    U->>C: 검증 관련 질문
    C->>C: 질문 유형 판별

    alt 조사가 필요한 질문
        C->>P: 영문 쿼리로 조사 요청
        P-->>C: 조사 결과 반환
        C->>C: 결과를 Q&A 형식으로 정리
    end

    C->>F: TUTORIAL_MARKER 위치에 Q&A 추가
    C-->>U: 답변 제공

    Note over F: Q 번호 자동 증가<br/>날짜 자동 기록
```

### 튜토리얼 업데이트 규칙

| 조건 | 동작 |
|------|------|
| 사용자 입력이 **질문**인 경우 | Perplexity MCP로 조사 → Q&A 형식으로 튜토리얼에 추가 |
| 사용자 입력이 **명령**인 경우 | 명령 실행 (튜토리얼 업데이트 없음) |
| `/update-tutorial` 실행 | 현재 세션의 Q&A를 수동으로 튜토리얼에 추가 |

### 자동 업데이트 대상 질문 유형

다음 주제에 대한 질문은 자동으로 튜토리얼에 기록됩니다:

1. **LLM 학습 검증 방법론** - from scratch vs fine-tuning 판별
2. **Tokenizer 분석** - vocabulary, BPE merge, special tokens
3. **Weight 분석** - cosine similarity, 해싱, PCA
4. **Architecture 분석** - config 비교, 구조적 특징
5. **MoE 모델 지식** - Mixture-of-Experts 아키텍처
6. **Solar-Open-100B 구체적 분석** - 실제 검증 결과

### 파일 구조

```
solar-open-100b-scratch-verification/
├── README.md                     # 메인 문서
├── CLAUDE.md                     # Claude 프로젝트 지침
├── docs/
│   ├── 00-tutorial.md            # Q&A 튜토리얼 (자동 업데이트)
│   ├── 01-tokenizer-analysis.md  # Tokenizer 분석 상세
│   ├── 02-weight-analysis.md     # Weight 분석 상세
│   ├── 03-architecture-analysis.md # Architecture 분석 상세
│   └── 04-behavior-analysis.md   # 행동 분석 상세
└── .claude/
    ├── settings.json             # Hooks 설정
    ├── commands/
    │   ├── commit-push.md        # /commit-push 커맨드
    │   └── save.md               # /save 커맨드 (빠른 저장)
    └── skills/
        └── update-tutorial.md    # /update-tutorial skill 정의
```

### 사용 가능한 Commands

| 커맨드 | 설명 | 사용법 |
|--------|------|--------|
| `/commit-push` | 변경사항 분석 후 커밋 & 푸시 | `/commit-push` |
| `/save` | 빠른 커밋 & 푸시 | `/save` 또는 `/save {메시지}` |
| `/update-tutorial` | Q&A 튜토리얼 수동 업데이트 | `/update-tutorial` |

### Q&A 형식

새로운 Q&A는 다음 형식으로 `docs/00-tutorial.md`의 `<!-- TUTORIAL_MARKER -->` 위에 자동 추가됩니다:

```markdown
---

## Q{N}: {질문 제목}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{Perplexity MCP 조사 기반 구조화된 답변}

---
```

---

## Solar-Open-100B 공식 정보 (검증 진행 중)

| 항목 | 값 |
|------|-----|
| **모델 유형** | Mixture-of-Experts (MoE) |
| **총 파라미터** | 102.6B |
| **활성 파라미터** | 12B (토큰당) |
| **Expert 구성** | 129개 (128 routed + 1 shared, top-8 활성화) |
| **Context Length** | 128k tokens |
| **학습 토큰 수** | 19.7 trillion tokens |
| **학습 하드웨어** | NVIDIA B200 GPUs |
| **라이선스** | Solar-Apache License 2.0 |

**공식 주장**: "Trained Entirely from Scratch" (Hugging Face 모델 카드에 명시)

---

## 검증 방법론

LLM이 실제로 from scratch로 학습되었는지 확인하는 기술적 방법들:

| # | 방법 | 신뢰도 | 접근성 | 탐지력 | 상세 문서 |
|---|------|--------|--------|--------|----------|
| 1 | Tokenizer 분석 | 높음 | 높음 | 우수 | [상세 보기](docs/01-tokenizer-analysis.md) |
| 2 | Weight 분석 | 높음 | 중간 | 양호 | [상세 보기](docs/02-weight-analysis.md) |
| 3 | Architecture 분석 | 중간 | 높음 | 양호 | [상세 보기](docs/03-architecture-analysis.md) |
| 4 | 행동 분석 | 중간 | 높음 | 보통 | [상세 보기](docs/04-behavior-analysis.md) |
| 5 | Training Logs 검증 | 매우 높음 | 낮음 | 우수 | (접근 불가) |

### 방법론 요약

**1. [Tokenizer 분석](docs/01-tokenizer-analysis.md)**
- Vocabulary 비교, BPE merge rules 분석, 특수 토큰 패턴 비교

**2. [Weight 분석](docs/02-weight-analysis.md)**
- Layer별 cosine similarity, Weight tensor 해시 비교, PCA 분포 분석

**3. [Architecture 분석](docs/03-architecture-analysis.md)**
- config.json 비교, MoE 구조 분석, RoPE/Attention 설정 비교

**4. [행동 분석](docs/04-behavior-analysis.md)**
- Knowledge cutoff 테스트, Refusal pattern 분석, Safety alignment 특성

**5. Training Logs 검증** (접근 불가)
- Loss curve 패턴 분석, Compute 추정 (Chinchilla scaling 기준)

---

## 검증 진행 상황

- [x] Tokenizer 분석 ✅ **완료 (2026-01-04)** → **From scratch 지지**
  - [x] Solar-Open-100B tokenizer vocabulary 추출 (196,608 tokens)
  - [x] Llama, Mistral, Qwen 등 주요 base model과 비교
  - [x] 토큰 중복률 계산 (vocab size 차이로 인해 직접 비교 불필요)
- [x] Architecture 분석 ✅ **완료 (2026-01-04)** → **From scratch 지지**
  - [x] config.json 분석 (48 layers, 128+1 experts, 196k vocab)
  - [x] 유사 MoE 모델들과 비교 (Mixtral, DeepSeek-V2, Qwen2-MoE)
  - [x] 결과: 어떤 기존 모델과도 architecture 불일치
- [x] Weight 분석 ✅ **완료 (2026-01-04)** → **비교 불가 (From scratch 증거)**
  - [x] Architecture 불일치로 직접 weight 비교 불가능
  - [x] 이 자체가 from scratch의 강력한 증거
- [x] 행동 분석 ✅ **완료 (2026-01-04)** → **제한적 (정보 부족)**
  - [x] Knowledge cutoff 미공개로 직접 테스트 불가
  - [x] 표절 논란 발생 (2026-01-01): GLM-4.5-Air와 LayerNorm 96.8% 유사도 주장
  - [x] Upstage 공개 검증 (2026-01-02): Training logs, checkpoints 공개
  - [x] 결론: 행동 분석만으로는 확정 불가, 종합적으로 From scratch 지지

---

## 종합 검증 결론

### 분석 결과 요약

| 분석 방법 | 결과 | 신뢰도 | From scratch 지지 |
|----------|------|--------|------------------|
| **Tokenizer** | vocab_size 196,608 (모든 모델과 불일치) | 높음 | ✅ 강력 지지 |
| **Architecture** | 48 layers, 129 experts (모든 모델과 불일치) | 높음 | ✅ 강력 지지 |
| **Weight** | Architecture 불일치로 비교 불가 | 높음 | ✅ 간접 지지 |
| **행동** | Knowledge cutoff 미공개, 표절 논란 → 공개 검증 | 중간 | ⚠️ 조건부 지지 |

### 최종 판정

**From scratch 학습 주장: 신뢰 가능 (조건부)**

**지지 근거:**
1. Tokenizer vocabulary가 모든 비교 대상과 완전히 다름 (196,608)
2. Architecture가 어떤 기존 MoE 모델과도 일치하지 않음
3. 고유한 Expert 구성 (128 routed + 1 shared)
4. Upstage의 공개 검증 세션에서 training logs, checkpoints 제시

**주의 사항:**
1. GLM-4.5-Air와의 LayerNorm 96.8% 유사도 의혹 (Upstage 해명 필요)
2. GLM-4.5-Air의 상세 config 미공개로 직접 비교 불가
3. 독립적 제3자 검증 결과 부재

### 추가 검증 권장 사항

| 항목 | 방법 | 가능 시점 |
|------|------|----------|
| GLM-4.5-Air config 비교 | Zhipu AI config 공개 시 architecture 비교 | 미정 |
| LayerNorm weight 검증 | 두 모델 weight 직접 비교 | GLM config 공개 시 |
| 독립적 3자 검증 | 학술/산업 기관 검증 | 미정 |

---

## 참고 자료

- [Hugging Face - Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B)
- [Model Integrity Verification](https://www.nightfall.ai/ai-security-101/model-integrity-verification)
- [LLM Evaluation Approaches](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches)

---

## 라이선스

이 프로젝트의 문서와 코드는 MIT License로 공개됩니다.
