---
title: "LLM-DNA 개요"
type: concept
tags: [llm-dna, model-lineage, iclr-2026]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: [raw/2026-04-26_phylogenetic-methods.md]
---

# LLM-DNA 개요

LLM-DNA는 ICLR'26 Oral 논문 ["LLM DNA: Tracing Model Evolution via Functional Representations"](https://openreview.net/pdf?id=UIxHaAqFqQ)에서 제안된 방법으로, 임의의 LLM에 대해 **저차원·학습 불필요(training-free)·아키텍처 무관**의 functional fingerprint(=DNA 벡터)를 추출하여 모델 간 계통 관계를 추적한다.

## 핵심 아이디어

생물학의 DNA처럼 모델의 본질을 압축한 표현이 존재해야 한다는 가설에서 출발한다. 저자들은 이 표현이 두 가지 수학적 성질([[inheritance-and-determinism]])을 만족함을 증명했다 — **inheritance**(부모-자식 관계가 DNA 거리에 보존됨)와 **genetic determinism**(같은 모델에서 추출한 DNA는 결정론적). 이 성질들이 [[phylogenetic-tree]] 구축의 정당성을 부여한다.

## 왜 functional representation인가

이전 방법들(weight 직접 비교, architecture 비교)은 두 가지 한계가 있었다. 첫째, 토크나이저나 아키텍처가 다르면 비교 자체가 불가능했다. 둘째, weight 공간은 너무 크고 노이즈가 많아 직접 비교가 불안정했다. LLM-DNA는 이를 우회해서 **모델의 출력 행동(functional output)** 만 본다 — 같은 입력을 주고 출력 임베딩을 모은 뒤 [[random-projection]]으로 압축한다. 이 방식은 [[architecture-analysis]]가 막히는 custom 아키텍처(예: [[ax-k1]]의 AXK1)에도 그대로 적용된다.

## 실험 규모

305개 LLM에 대해 검증을 수행했고, 기존 연구의 부분집합 결과와 정합성을 확인했으며 동시에 **공식 문서화되지 않은 관계**를 다수 발견했다. 시간순 진화, encoder-decoder→decoder-only 전환, 모델 패밀리별 진화 속도 차이가 트리에 자연스럽게 드러난다.

## 본 프로젝트에서의 활용

본 프로젝트는 [[dokpamo-project]] 차기 라운드 진출 3종([[solar-open-100b]], [[k-exaone-236b]], [[ax-k1]])의 functional 계통을 추출하여, [[llama-3-family]]·[[qwen-25-family]]·[[glm-family]]·[[mixtral-8x7b]]·[[deepseek-v25]] reference 클러스터와의 거리를 측정한다. [[neighbor-joining]] 알고리즘으로 트리를 구축하면 from scratch 주장의 강한 정황 증거가 된다.

## 관련 페이지

- 알고리즘 상세: [[dna-extraction-pipeline]]
- Python 패키지: [[llm-dna-package]]
- 거리 행렬 → 트리: [[neighbor-joining]], [[phylogenetic-tree]]
- 본 프로젝트 인프라: [[sagemaker-spot-training]]
