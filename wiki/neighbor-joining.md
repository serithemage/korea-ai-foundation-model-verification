---
title: "Neighbor-Joining 알고리즘"
type: concept
tags: [phylogenetics, algorithm, saitou-nei]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: [raw/2026-04-26_phylogenetic-methods.md]
---

# Neighbor-Joining 알고리즘

Saitou & Nei (1987)가 제안한 거리 기반 [[phylogenetic-tree]] 구축 알고리즘. **울트라메트릭 가정 없이** 거리 행렬에서 unrooted 트리를 구축한다. 본 프로젝트의 LLM-DNA 벡터 분석에 가장 적합한 표준 도구.

## 작동 방식

`n × n` 거리 행렬 `D`로 시작.

### Q-matrix
각 반복에서 수정 거리 행렬 Q를 계산:
```
Q(i,j) = (n-2) · d(i,j) − Σ_k d(i,k) − Σ_k d(j,k)
```
Q를 최소화하는 쌍 `(i,j)`을 "이웃"으로 선택한다. 단순히 가장 가까운 쌍이 아니라, **다른 모든 노드에서 떨어져 있는 정도**를 보정한 값이다.

### 가지 길이 계산
새 내부 노드 `u`에 대해:
```
δ(i,u) = ½·d(i,j) + 1/(2(n−2))·(Σ d(i,k) − Σ d(j,k))
δ(j,u) = d(i,j) − δ(i,u)
```

### 거리 갱신
나머지 모든 노드 `k`에 대해 새 노드와의 거리:
```
d(u,k) = ½·(d(i,k) + d(j,k) − d(i,j))
```

이를 반복해 노드 2개만 남으면 직접 연결, 트리 완성.

## 복잡도

- 시간: `O(n^3)` (표준 구현)
- Rapid NJ, Fast NJ 등 최적화 변형은 `O(n^2)`로 개선

본 프로젝트 대상 모델 수는 ~10개라 표준 구현으로도 즉시 끝난다.

## 핵심 보장

**거리 행렬이 가법적(additive)이면 NJ는 정확한 트리 위상을 복원**한다. 또한 거리 추정 오차가 가장 짧은 가지의 절반보다 작으면 위상 안정성이 보장된다 — 노이즈 강건성이 높다는 의미.

[[upgma]]와 비교한 핵심 차이: NJ는 진화율 균일성을 가정하지 않는다. ML 모델 간 거리는 어떤 쌍은 fine-tune 관계로 가깝고 어떤 쌍은 from scratch라 멀므로, 균일 가정을 하지 않는 NJ가 거의 항상 정답이다.

## 본 프로젝트 사용

```python
from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, DistanceMatrix
import numpy as np

# DNA 벡터들을 모아 거리 행렬 계산 (cosine 추천: distance-metrics 참조)
dna_vectors = {model: np.load(f"out/rand/{model}/dna_vector.npy") for model in models}
dm = DistanceMatrix(names=list(dna_vectors), matrix=lower_triangular_distances)
tree = DistanceTreeConstructor().nj(dm)
Phylo.write(tree, "lineage.nwk", "newick")
```

Newick 포맷을 [iTOL](https://itol.embl.de/)에 업로드하면 인터랙티브 시각화 가능.

## 출력은 unrooted

NJ는 진화 방향 정보를 주지 않는다. 부모-자식 추정이 필요하면:
- **Outgroup rooting**: 명확히 무관한 모델(예: GPT-2)을 추가해 거기를 루트로
- **Midpoint rooting**: 가장 먼 두 잎의 중점을 루트로 (분자 시계 가정)

본 프로젝트는 from scratch 여부 판정이 목적이므로 unrooted 트리만으로 충분 — "[[ax-k1]]이 어느 군에 가까운가"가 핵심 질문.
