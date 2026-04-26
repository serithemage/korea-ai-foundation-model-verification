---
title: "Upstage Solar-Open-100B"
type: entity
tags: [korean-llm, moe, dokpamo, upstage]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Upstage Solar-Open-100B

[[dokpamo-project]] 1차 평가 3위로 [Phase 2 진출](../README.md). HuggingFace: [upstage/Solar-Open-100B](https://huggingface.co/upstage/Solar-Open-100B). 2025-12-10 공개.

## 핵심 사양

| 항목 | 값 |
|------|----|
| Architecture | `SolarOpenForCausalLM` (`solar_open` model_type) |
| 총 파라미터 | 102B |
| 활성 파라미터 (MoE) | ~7B (8 of 128 routed + 1 shared per token) |
| Layers | 48 |
| Hidden size | 4,096 |
| Attention heads | 64 (KV heads 8 → GQA) |
| **vocab_size** | **196,608** |
| Routed experts | 128 |
| Shared experts | 1 |
| Experts per token | 8 |
| Intermediate size | 10,240 (dense) / 1,280 (MoE) |
| RoPE θ | 1,000,000 |
| Max context | 131,072 |
| dtype | bfloat16 |

vocab_size 196,608은 본 프로젝트 검증 대상 중 가장 크고, 어떤 reference 모델과도 일치하지 않는다 ([[tokenizer-analysis]] 강력 from scratch 증거).

## 검증 사용 모델 (LLM-DNA)

본 프로젝트 [[dna-extraction-pipeline]]에서 ml.p5.48xlarge 인스턴스로 fp16 추론 (~200GB 메모리 필요).

## "From Scratch" 논란 정리

[[from-scratch-debate]]의 시발점이 된 모델. 2026-01-01 PsionicAI 고석현 CEO가 [[glm-family]] (Zhipu) 코드 차용 의혹 제기 → 1월 2일 강남역 인근에서 [[layernorm-fingerprint-fallacy]] 사례로 알려진 공개 검증 세션 → 고석현 사과문(전적 철회는 아님).

상세: [Solar-Open-100B 검증 보고서](../README.md#1-upstage-solar-open-100b-) 참조.

## 차별화 요소

- **Training logs 공개**: [YouTube Live](https://www.youtube.com/live/2YY9aAUSo_w)에서 라이브 세션. Phase 1~5 모델 중 유일하게 학습 과정을 외부에 노출.
- **129 experts (128+1)**: 본 프로젝트 검증 대상 중 가장 일반적인 구성. [[k-exaone-236b]]도 동일 구조.

## LLM-DNA 분석 가설

만약 진짜 from scratch라면 reference 클러스터([[llama-3-family]], [[qwen-25-family]], [[glm-family]])에서 멀리 떨어진 위치에 단독 또는 [[k-exaone-236b]]·[[ax-k1]]과 함께 한국 모델 clade를 형성할 것으로 예상.

만약 GLM 기반 fine-tune이라면 [[glm-family]] (특히 GLM-5-FP8) 근처에 cluster될 것. 단 vocab_size 차이로 weight 직접 비교는 불가능하다는 점이 반론의 근거.

## 관련 페이지

- [[from-scratch-debate]] — 논란 전체 맥락
- [[layernorm-fingerprint-fallacy]] — LayerNorm 유사도 분석의 한계
- [[glm-family]] — 비교 reference
- [[dokpamo-project]] — 사업 맥락
