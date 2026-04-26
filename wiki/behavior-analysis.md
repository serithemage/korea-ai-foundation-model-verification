---
title: "Behavior 분석"
type: concept
tags: [verification, methodology, behavior, knowledge-cutoff]
status: active
confidence: medium
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Behavior 분석

모델의 출력 행동(refusal pattern, knowledge cutoff, 응답 스타일)을 base 모델과 비교해 파생 관계를 추정. 가장 간접적이지만 직접 실행만 가능하면 즉시 시도 가능.

본 프로젝트 상세 문서: [`docs/04-behavior-analysis.md`](../docs/04-behavior-analysis.md).

## 분석 항목

### Knowledge Cutoff
"What's the latest event you know about?"같은 prompt로 cutoff 추정. 두 모델이 동일 cutoff면 같은 데이터로 학습 의심. 단 post-training으로 cutoff를 수정할 수 있어 결정적이지 않음.

### Refusal Pattern
민감한 prompt에 대한 거부 문구 패턴. 같은 RLHF/DPO 데이터셋을 쓴 모델은 비슷한 거부 형식 사용. 예: Llama 계열 "I can't assist with that..." 스타일.

### 출력 스타일
- 응답 시작 패턴 ("Sure!" / "Certainly,")
- 마크다운 사용 빈도
- Chain-of-thought 자발성

이런 미묘한 스타일이 base의 흔적 가능성.

## 한계

- **Post-training으로 변경 가능**: SFT나 RLHF로 행동을 완전히 바꿀 수 있음
- **직접 실행 환경 필요**: API 또는 로컬 GPU. 본 프로젝트는 [[ax-k1]]·[[k-exaone-236b]]·[[solar-open-100b]] 모두 직접 추론 가능 환경 없어 미수행
- **subjective**: 정량화 어려움

## LLM-DNA와의 관계

[[llm-dna-overview]]는 본질적으로 **자동화된 behavior 분석**이다. 사람이 prompt 만들고 답 비교하는 정성적 접근을 대신해, 100개 random prompt를 일괄 입력하고 출력 임베딩을 차원 축소한다.

차이:
- Manual behavior 분석: 해석 가능하지만 적은 prompt, 주관적
- LLM-DNA: 정량적·자동화·확장성 좋지만 트리만 보면 "왜" 가까운지 직관 없음

본 프로젝트는 두 접근을 보완적으로 사용한다 — LLM-DNA로 1차 distance map → 의심 쌍에 대해 manual probe로 deeper inspection.

## 관련 페이지

- [[llm-dna-overview]] — 자동화된 behavior 분석
- [[weight-analysis]] — 정적 비교
- [[from-scratch-debate]] — behavior 분석이 결정적 증거가 못 되는 이유
