# 튜토리얼: Q&A 형식 학습 기록

이 문서는 클로드 스킬을 이용해서 국가 AI 파운데이션 모델 검증 프로젝트(구 Solar-Open-100B 검증 프로젝트)를 진행하면서 던진 질문과 답변을 주제별로 정리한 것입니다.

## 목차

### 1. 기초 개념 ([01-기초개념.md](01-기초개념.md))

프로젝트의 배경과 검증 방법론의 기본 개념을 다룹니다.

- **Q0**: 왜 "from scratch" vs "fine-tuning" 논란이 발생하나요?
- **Q1**: LLM이 "from scratch"로 학습되었는지 어떻게 검증할 수 있나요?

### 2. Tokenizer 분석 ([02-tokenizer-분석.md](02-tokenizer-분석.md))

Tokenizer 기반 검증 방법과 실제 분석 결과를 다룹니다.

- **Q2**: Tokenizer 분석이란 무엇이고, 어떻게 from scratch 여부를 판별하나요?
- **Q5**: Solar-Open-100B의 Tokenizer 분석 결과는 어떠한가요?
- **Q6**: Fine-tuning 의심 증거들에 대한 반론은 무엇인가요?
- **Q10**: vocab_size 일치가 왜 중요한가요?

### 3. Weight 및 Architecture 분석 ([03-weight-architecture-분석.md](03-weight-architecture-분석.md))

Weight 비교와 architecture 분석 방법을 다룹니다.

- **Q3**: Weight 분석이란 무엇이고, 어떻게 from scratch 여부를 판별하나요?
- **Q4**: Cosine Similarity 분석이란 무엇이고, LLM weight 비교에 어떻게 활용되나요?
- **Q7**: Solar-Open-100B의 Weight 분석이 가능한가요? Architecture 비교 결과는?
- **Q12**: 왜 많은 대규모 MoE 모델들이 48 레이어와 129 experts를 사용하나요?

### 4. 사례 연구 ([04-사례연구.md](04-사례연구.md))

실제 모델들의 검증 사례와 논란을 다룹니다.

- **Q8**: Solar-Open-100B의 행동 분석 결과는? (표절 논란과 공개 검증)
- **Q9**: HyperCLOVAX-SEED-Think-32B는 from scratch인가요?

### 5. 방법론 평가 ([05-방법론-평가.md](05-방법론-평가.md))

검증 방법론에 대한 비판적 검토와 학술 연구 결과를 다룹니다.

- **Q11**: 이 프로젝트의 검증 방법론에 대한 비판과 학술 연구 결과는?

---

## 학습 순서 추천

1. **입문자**: 01 → 02 → 04 순서로 읽으면 프로젝트의 맥락을 이해할 수 있습니다.
2. **기술 심화**: 03 → 05 순서로 읽으면 검증 기법의 세부 사항을 파악할 수 있습니다.
3. **빠른 참조**: 각 파일의 특정 Q&A만 필요할 때 직접 접근하세요.

---

## 새로운 Q&A 추가 안내

새로운 Q&A는 `/update-tutorial` 스킬을 통해 주제에 맞는 파일에 자동 추가됩니다.

| 주제 | 대상 파일 |
|------|----------|
| from scratch 개념, 검증 방법 개요 | 01-기초개념.md |
| Tokenizer, vocabulary, merge rules | 02-tokenizer-분석.md |
| Weight, cosine similarity, architecture | 03-weight-architecture-분석.md |
| 특정 모델 검증 결과, 논란 사례 | 04-사례연구.md |
| 방법론 비판, 학술 연구, 개선 방향 | 05-방법론-평가.md |
