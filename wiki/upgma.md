---
title: "UPGMA 알고리즘"
type: concept
tags: [phylogenetics, algorithm, hierarchical-clustering]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: [raw/2026-04-26_phylogenetic-methods.md]
---

# UPGMA 알고리즘

Unweighted Pair Group Method with Arithmetic Mean. 가장 단순한 거리 기반 [[phylogenetic-tree]] 구축 알고리즘으로, 평균 연결 계층적 군집화(average linkage hierarchical clustering)와 본질적으로 동일하다.

## 작동 방식

각 반복:
1. 거리 행렬에서 가장 가까운 쌍 `(s, t)`를 찾는다
2. 두 군집을 합친다
3. 새 군집과 나머지 군집들 간 거리는 멤버 간 평균:
   ```
   d(s ∪ t, k) = (|s|·d(s,k) + |t|·d(t,k)) / (|s| + |t|)
   ```
4. 모두 합쳐질 때까지 반복

출력은 **rooted dendrogram** — 모든 잎이 루트와 등거리.

## 가정: 울트라메트릭

UPGMA는 거리 행렬이 울트라메트릭이라 가정한다 — 모든 잎이 루트로부터 같은 거리. 이는 **분자 시계** 가정과 동치 — 모든 가지에서 진화 속도가 일정.

이 가정이 성립하지 않으면 트리 위상이 왜곡된다. 가까워야 할 모델이 떨어지고 먼 모델이 묶이는 식.

## 본 프로젝트에서는 부적합

ML 모델은 균일 진화율을 거의 만족하지 않는다. 예시:

- [[solar-open-100b]] vs [[k-exaone-236b]]: 둘 다 from scratch라 reference에서 거리가 비슷할 수 있음
- 한 reference fine-tune된 가상의 모델과 from scratch 모델: 거리 차이가 극단적
- 같은 family 내 사이즈만 다른 변종 ([[qwen-25-family]] 7B vs 72B): 함수적으로 가깝지만 학습량 차이로 거리가 균등하지 않음

이런 비균일성을 [[neighbor-joining]]은 자연스럽게 수용하지만 UPGMA는 강제로 균일화한다.

## 그래도 쓸 만한 경우

- 빠른 탐색용 (시간 복잡도 `O(n^2)`, NJ는 `O(n^3)`)
- "큰 그룹이 어디에 있나" 정도의 1차 클러스터링
- rooted 구조가 필요하고 분자 시계가 합리적인 경우

본 프로젝트도 sanity check로 UPGMA를 먼저 돌려 NJ 결과와 비교해 보는 정도는 가능.

## 도구

```python
from scipy.cluster.hierarchy import linkage, dendrogram
# UPGMA = method='average'
Z = linkage(condensed_distance_vector, method='average')
dendrogram(Z, labels=model_names)
```

또는 `Bio.Phylo.TreeConstruction.DistanceTreeConstructor().upgma(dm)`.

## 핵심: NJ를 기본, UPGMA는 비교용

[[neighbor-joining]]을 기본 알고리즘으로 사용하고, UPGMA는 결과의 강건성 확인을 위한 보조 도구로만 사용한다.
