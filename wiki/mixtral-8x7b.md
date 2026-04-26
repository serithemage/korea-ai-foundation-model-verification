---
title: "Mistral Mixtral-8x7B"
type: entity
tags: [reference-model, mistral, moe]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Mistral Mixtral-8x7B

Mistral AI의 sparse MoE base 모델. 본 프로젝트에서 **MoE 구조 baseline**으로 사용. HuggingFace: [mistralai/Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1). 2023-12-01 공개.

## 핵심 사양

| 항목 | 값 |
|------|----|
| Architecture | MixtralForCausalLM |
| 총 파라미터 | 47B (8 experts × 7B basis) |
| 활성 파라미터 | ~13B (2 experts per token) |
| Layers | 32 |
| Hidden size | 4,096 |
| 인스턴스 | ml.p5.4xlarge 4-bit |
| 라이선스 | Apache 2.0 |

## Reference로 선택한 이유

본 프로젝트 검증 대상 3종이 모두 MoE이므로, **MoE 구조 자체가 만드는 functional similarity**를 calibration할 baseline이 필요. Mixtral-8x7B는:

- 가장 보편적인 MoE 모델 (Apache 2.0 + 광범위한 채택)
- 8 experts (한국 모델 128~192보다 훨씬 적음 → 다른 점 부각)
- 다국어 지원 (영·프·이·독·스)

만약 MoE 구조 자체가 DNA에 강하게 반영된다면 Mixtral과 한국 MoE 모델들이 Llama/Qwen dense 모델보다 가까워야 한다. 이를 가설 검증 가능.

## LLM-DNA 분석 의의

- Mixtral과 [[solar-open-100b]] / [[k-exaone-236b]] / [[ax-k1]] 거리가 가까움 → MoE 구조 영향 큼 (이 경우 trees 해석에 주의)
- Mixtral과 한국 MoE가 멀고, 한국 모델끼리는 가까움 → 한국군 자체 시그널이 강함
- 모든 MoE가 Llama/Qwen dense보다 가까움 → MoE라는 추가 차원이 트리에 드러남

## 차별화

Mixtral-8x7B는 [[deepseek-v25]] 236B나 한국 100B+ 모델 대비 **훨씬 작은 MoE**라는 점이 calibration에 유리. 같은 MoE라도 sparse vs heavy MoE의 functional difference를 가시화할 수 있다.

## 함정

Instruct 변종(Mixtral-8x7B-Instruct-v0.1)이 더 많이 사용되지만 본 프로젝트는 **base** 모델 사용 (RLHF 영향을 배제하기 위해).
