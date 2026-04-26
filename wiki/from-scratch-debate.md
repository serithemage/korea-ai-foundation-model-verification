---
title: "From Scratch 논란"
type: concept
tags: [policy, from-scratch, controversy, dokpamo]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# From Scratch 논란

[[dokpamo-project]] 사업의 핵심 정의 이슈. "from scratch 학습"이란 무엇이고, 어디까지 허용되는가에 대한 명시적 정량 기준이 정부에 의해 발표되지 않은 상태.

## 정의 시도

### 가장 엄격한 정의
**Random init weight + 자체 데이터 + 자체 코드 + 자체 tokenizer**. 모든 컴포넌트가 외부 모델과 무관해야.

### 실용적 정의
Weight init이 random이고, training pipeline이 자체적이면 OK. Tokenizer·data·code는 부분적 외부 차용 허용 (대부분의 LLM이 SentencePiece, transformers, public corpora 사용).

### 가장 관대한 정의
"Korean control 하에 학습된 모델". 외부 base를 fine-tune해도 한국 조직이 control하면 OK. 정부 가이드라인 자체가 모호하므로 실질적으로 이 정의가 적용될 가능성도.

## 주요 사건

### Solar-GLM 의혹 (2026-01-01)
PsionicAI 고석현이 [[solar-open-100b]]의 [[layernorm-fingerprint-fallacy]] 분석으로 [[glm-family]] 차용 의혹 제기 → Upstage 공개 검증 → 고석현 사과문 (단, 핵심 의혹 철회 X).

### HyperCLOVAX Vision/Audio Encoder (2026-01-05)
NAVER가 Qwen2.5 Vision/Audio Encoder 재사용 사실 공식 인정. Vision Encoder cosine 유사도 99.51%. 단 NAVER는 "Text Decoder는 100% 자체"라고 주장하며, 정부 가이드라인이 VLM 컴포넌트별 from scratch 요건을 명시하지 않은 점을 근거로.

### Tokenizer 분석 (2026-01-05)
HyperCLOVAX-SEED tokenizer가 [[qwen-25-family]]의 BPE merge rules 처음 20개와 완전 일치. 본 프로젝트 tutorial Q11. 추가 검증 필요한 상태.

## 정부 입장

배경훈 부총리: "혁신은 투명하고 엄정한 검증 속에서 단련된다 ... 한국 AI가 더 높이 도약하기 위해 반드시 거쳐야 할 과정". 검증 과정 자체를 사업의 정당성 강화 요소로 프레이밍. **그러나 4월 현재까지 정량 기준은 미발표**.

## 학술적 관점

[[model-provenance-testing]] 학술 연구는 다음을 시사:
- Architecture 일치만으로는 파생 증거 불충분 (Yi-Llama 사례)
- Black-box output similarity가 90~95% precision으로 가장 신뢰 가능
- LayerNorm 등 단일 통계량 분석은 함정 ([[layernorm-fingerprint-fallacy]])

## 본 프로젝트의 입장

본 프로젝트는 **단정 회피 + 다중 evidence cross-validation** 원칙. 단일 metric으로 결론 내지 않고:
1. [[tokenizer-analysis]] (vocab + BPE merge)
2. [[architecture-analysis]] (config 비교)
3. [[llm-dna-overview]] (functional behavior)

세 신호가 모두 같은 방향이면 강한 결론, 어긋나면 추가 조사 권장.

## 향후 리스크

- **차기 라운드 (2026 하반기) 평가**에서 또 from scratch 논란 발생 가능성
- 정부가 명시 기준 발표 시 기존 모델 재평가 필요
- 학술 검증과 정치적 평가의 lag — Yi-Llama 사례는 학계 결론까지 ~6개월

## 관련 페이지

- [[dokpamo-project]] — 사업 맥락
- [[layernorm-fingerprint-fallacy]] — 핵심 분석 오류 사례
- [[model-provenance-testing]] — 학술적 검증 기준
- [[solar-open-100b]] — 논란의 시발 모델
