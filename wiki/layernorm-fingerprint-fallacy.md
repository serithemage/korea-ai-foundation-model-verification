---
title: "LayerNorm Fingerprint Fallacy"
type: concept
tags: [verification, fallacy, layernorm, solar, controversy]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# LayerNorm Fingerprint Fallacy

[[solar-open-100b]] from scratch 논란의 출발점이 된 분석 오류. **LayerNorm parameter cosine similarity로 두 모델의 차용 여부를 판단할 수 없다**는 결론.

## 사건 경과

2026-01-01 PsionicAI 고석현 CEO가 X에 다음 주장 게시:
- "Solar의 LayerNorm parameter가 GLM과 96.8% cosine similarity"
- 따라서 Solar는 [[glm-family]] 차용

이튿날 1월 2일 Upstage가 강남역 인근에서 공개 검증 세션 개최. 이후 hyunwoongko가 [solar-vs-glm-vs-phi](https://github.com/hyunwoongko/solar-vs-glm-vs-phi) 저장소에서 독립 검증 → 분석 결함 발견 → 1월 말 고석현 사과문 게시.

## 왜 LayerNorm은 fingerprint가 될 수 없는가

### 1. Init 편향
LayerNorm weight는 **모두 1.0으로 초기화**된다. 학습 후에도 평균 ~1.0 근처에 머무는 경향. 두 모델이 from scratch라도 LayerNorm vector는 비슷한 방향(1, 1, 1, ...)을 가리킬 수밖에 없다 — 즉 cosine similarity가 자연히 높게 나옴.

### 2. 같은 모델 내 다른 layer도 0.99
hyunwoongko의 분석에서 한 모델의 layer 0와 layer 47 LayerNorm cosine이 0.99 수준. 이게 "같은 모델이라" 라는 게 아니라 **init 분포의 부산물**. 모델 간 비교 0.96은 모델 내부 비교 0.99보다 오히려 낮은 셈.

### 3. Centered cosine으로 보면
평균 오프셋(≈1.0)을 빼고 cosine 계산하면 → 모델 간 LayerNorm 유사도가 거의 0으로 하락. 진짜 파라미터 정렬은 없었다는 의미.

## 교훈

### Statistical signature ≠ provenance
높은 통계 유사도가 자동으로 차용을 의미하지 않는다. **init 편향·구조적 제약을 항상 통제**해야 한다.

### 적절한 baseline 필요
"두 모델이 96.8% 유사" 만으로는 부족. **무작위 두 모델은 얼마나 유사한가** baseline이 필요. 그 baseline이 95%면 96.8%는 무의미.

### Cherry-picked layer 위험
LayerNorm처럼 init 편향이 큰 specific tensor만 골라 비교하면 fallacy 빠지기 쉬움. 전체 weight 분포 또는 functional behavior를 봐야 한다.

## LLM-DNA와의 관계

[[llm-dna-overview]]는 이런 fallacy를 구조적으로 회피한다:
- **무작위 prompt** 사용 (cherry-pick 불가)
- **Random projection** (단일 차원 의존성 제거)
- **305 모델 검증** (baseline 풍부)
- 통계적 boundary가 명확 (논문에 distance threshold 제시)

## 본 프로젝트 시사

본 프로젝트가 [[tokenizer-analysis]] + [[architecture-analysis]] + [[llm-dna-overview]] **cross-validation**을 강조하는 이유. 단일 metric을 cherry-pick해 단정하지 않고, 다중 evidence가 모두 같은 방향을 가리킬 때만 결론 도출.

## 관련 페이지

- [[solar-open-100b]] — 사건 당사자
- [[from-scratch-debate]] — 사건의 광범위한 맥락
- [[weight-analysis]] — centered cosine 등 올바른 weight 비교 방법
- [[model-provenance-testing]] — 학술적 best practice
