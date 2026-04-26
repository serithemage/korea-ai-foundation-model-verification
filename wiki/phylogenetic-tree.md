---
title: "Phylogenetic Tree (계통수)"
type: concept
tags: [phylogenetics, distance-based, lineage]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: [raw/2026-04-26_phylogenetic-methods.md]
---

# Phylogenetic Tree (계통수)

생물학에서 종 간 진화 관계를 표현하기 위해 개발된 트리 구조. 본 프로젝트는 LLM 모델 간 functional 거리 행렬에서 계통수를 구축해 [[llm-dna-overview]] DNA 벡터의 의미를 시각화한다.

## 입력: 거리 행렬

`n × n` 대칭 행렬, 대각이 0. 각 원소 `d(i,j)`는 두 모델 간 비유사도. 본 프로젝트는 [[dna-extraction-pipeline]]이 생성한 128차원 DNA 벡터들 사이의 [[distance-metrics]] 적용으로 행렬을 만든다.

## 알고리즘 선택

| 알고리즘 | 가정 | 적합 상황 |
|---------|------|----------|
| [[neighbor-joining]] | 가법성(additivity), 일정 진화율 가정 X | **대부분의 ML 모델 비교** |
| [[upgma]] | 울트라메트릭, 분자 시계 (모든 잎이 루트와 등거리) | 같은 시점에 같은 속도로 진화한 경우만 |

NJ는 unrooted 트리를, UPGMA는 rooted 덴드로그램을 출력한다. ML 모델은 진화 속도가 균일하지 않으므로 (어떤 모델은 fine-tune만, 어떤 모델은 from scratch) NJ가 거의 항상 적절하다.

## 출력 해석

- **branch length**: 두 노드 간 functional 거리. 길수록 다름.
- **clade(가지)**: 공통 조상에서 갈라진 그룹. 같은 clade의 모델들은 functional 유사성이 높음.
- **rotation 자유**: 가지를 회전해도 위상은 동일 — 좌우 순서는 의미 없음.
- **rooted vs unrooted**: NJ는 unrooted (방향성 없음). 루트가 필요하면 outgroup 모델(예: GPT-2 같은 명확히 무관한 reference)을 추가하거나 midpoint rooting 적용.

## 본 프로젝트의 활용 시나리오

[[solar-open-100b]], [[k-exaone-236b]], [[ax-k1]]의 DNA 벡터를 [[llama-3-family]]·[[qwen-25-family]]·[[glm-family]]·[[mixtral-8x7b]]·[[deepseek-v25]] reference와 함께 거리 행렬로 만들고 NJ를 돌리면, 한국 모델 3종이 어느 reference 클러스터에 가까운지 시각적으로 드러난다. 만약 from scratch라면 reference 어느 곳에서도 멀리 떨어진 위치에 자체 clade를 형성해야 한다.

## 도구

- **Python**: `Bio.Phylo.TreeConstruction` (BioPython), `scipy.cluster.hierarchy`, `ete3`
- **시각화**: `phyTreeViz`, `ete3` GUI, Newick 포맷 → iTOL 웹 viewer

## 한계

거리 행렬에 노이즈가 있으면 트리 위상이 흔들릴 수 있다. **bootstrap** (probe set 일부를 무작위로 빼고 여러 번 재구축) 으로 신뢰도 측정 권장. 핵심 분기(예: 한국 모델이 reference 어느 군과 가까운지)는 bootstrap >70%면 신뢰 가능.
