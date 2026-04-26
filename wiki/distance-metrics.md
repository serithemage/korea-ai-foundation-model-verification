---
title: "거리 메트릭 (고차원 임베딩용)"
type: concept
tags: [distance, embedding, metric]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: [raw/2026-04-26_phylogenetic-methods.md]
---

# 거리 메트릭 (고차원 임베딩용)

[[dna-extraction-pipeline]]가 생성한 128차원 DNA 벡터들 사이의 거리를 어떻게 정의할지가 [[neighbor-joining]] 트리 위상에 큰 영향을 준다.

## 4가지 후보

### Cosine distance (권장)
```
d_cos(u, v) = 1 − (u·v) / (||u|| ||v||)
```
두 벡터의 **각도**만 본다. 크기(magnitude)에 불변. 임베딩 공간에서 magnitude는 학습 과정의 부산물인 경우가 많아, 의미 있는 정보는 방향에 있다는 가정과 정합. 고차원 저주(curse of dimensionality)에도 상대적으로 강건.

본 프로젝트 권장 기본값.

### Euclidean (L2)
```
d_E(u, v) = √Σ(u_i − v_i)²
```
직관적이고 친숙하지만 고차원에서 모든 점이 점점 등거리에 가까워지는 문제 — **discriminative power가 떨어진다**. 또 차원별 스케일이 다르면 큰 스케일 차원이 거리를 지배.

DNA 벡터를 unit norm으로 정규화하면 cosine과 단조 관계가 되므로 결과적으로 비슷한 트리가 나오긴 한다.

### Manhattan (L1)
```
d_M(u, v) = Σ|u_i − v_i|
```
이산·범주형 특성에 강하지만 random projection 출력은 연속적이라 본 케이스에는 부적합.

### Correlation distance
```
d_corr(u, v) = 1 − Pearson(u, v)
```
방향뿐 아니라 affine 변환에도 불변(상수 더하기·곱하기 무시). 임베딩 패턴이 비슷하면 절대값이 달라도 가깝게 측정.

## 본 프로젝트 권장

**Cosine distance**를 기본. 이유:
1. [[random-projection]] 출력은 무작위 가우시안 사영이라 magnitude 정보가 의미 없음 → 각도만 봐야 함
2. 128차원도 충분히 고차원이라 Euclidean의 등거리화 문제 영향
3. [[llm-dna-overview]] 논문이 실험적으로 cosine을 검증

부수적으로 Euclidean·Correlation도 돌려 트리 위상이 일관적인지 확인하면 결과 신뢰도가 올라간다 (multi-metric robustness check).

## 정규화 주의

DNA 벡터를 cosine 거리에 넣기 전에 unit norm으로 정규화할 필요는 없다 (cosine 정의 자체가 정규화 포함). 그러나 만약 Euclidean을 쓴다면 사전 정규화가 거의 필수.

## scipy 사용법

```python
from scipy.spatial.distance import pdist, squareform
import numpy as np

X = np.stack([np.load(f"out/rand/{m}/dna_vector.npy") for m in models])  # (n, 128)
condensed = pdist(X, metric='cosine')  # 또는 'euclidean', 'correlation'
dm = squareform(condensed)             # n×n 정사각 행렬
```

이후 [[neighbor-joining]] 입력으로 전달.

## 함정: 차원 영향

차원 `k`가 너무 작으면 ([[random-projection]]에서 `dna_dim` 작게 잡으면) 모든 거리가 비슷해져 트리 위상이 불안정. 본 프로젝트는 `dna_dim=128`로 고정 — 305개 모델까지 안정적인 값.
