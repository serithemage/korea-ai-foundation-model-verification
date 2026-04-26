---
title: "Random Projection"
type: concept
tags: [dimensionality-reduction, johnson-lindenstrauss]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Random Projection

고차원 벡터를 무작위 가우시안 행렬 `R ∈ R^{d×k}`(평균 0, 분산 1/k)로 곱해 `k` 차원으로 사영하는 기법. [[dna-extraction-pipeline]]에서 functional matrix를 128차원 DNA 벡터로 압축할 때 사용된다.

## Johnson-Lindenstrauss 보조정리

핵심 이론적 보장은 JL 보조정리다 — `n`개 점을 차원 `O(log n / ε²)`로 사영해도 모든 쌍거리가 `(1±ε)` 배 안에서 보존된다. 즉 차원이 폭삭 줄어도 점들 간 상대적 위치 정보는 거의 손실 없다.

## PCA·SVD와 비교

[[llm-dna-package]]는 `pca`, `svd`, `random_projection` 세 옵션을 제공한다. 차이는:

- **PCA / SVD**: 데이터 분산을 가장 잘 보존하는 축을 학습한다. 대신 학습 데이터에 의존하므로, 비교하는 모델 셋이 바뀌면 사영 행렬도 다시 계산해야 한다 (= **inheritance 성질 깨짐**).
- **Random projection**: 데이터와 무관하게 시드만 고정하면 동일한 사영 행렬이 나온다. 새 모델이 추가돼도 기존 DNA 벡터를 재계산할 필요가 없다 (= **결정성·확장성 우수**).

LLM-DNA 논문은 이 데이터 독립성 때문에 random projection을 기본으로 채택했다.

## 실용적 고려

- 시드 고정 필수 (보통 42)
- 가우시안 R이 메모리에 한 번만 올라가면 되므로 빠름
- ε(왜곡 허용치)는 명시적으로 튜닝하지 않고, `k=128`로 충분히 작아 실용적으로 안정적

## 한계

차원이 너무 낮으면(예: k=8) JL 보장이 깨져 노이즈가 시그널을 덮을 수 있다. LLM-DNA의 128은 305개 모델 비교에 충분한 여유를 둔 값.
