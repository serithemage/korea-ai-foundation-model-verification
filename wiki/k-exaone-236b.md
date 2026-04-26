---
title: "LG K-EXAONE-236B-A23B"
type: entity
tags: [korean-llm, moe, dokpamo, lg-ai]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# LG K-EXAONE-236B-A23B

[[dokpamo-project]] 1차 평가 **1위**로 Phase 2 진출. HuggingFace: [LGAI-EXAONE/K-EXAONE-236B-A23B](https://huggingface.co/LGAI-EXAONE/K-EXAONE-236B-A23B). 2025-12-26 공개. EXAONE 4.0 시리즈의 메가급 변종.

## 핵심 사양

| 항목 | 값 |
|------|----|
| Architecture | `ExaoneMoEForCausalLM` (`exaone_moe` model_type) |
| 총 파라미터 | 236B |
| **활성 파라미터 (A23B)** | **23B** (8 experts active per token) |
| Layers | 48 |
| Hidden size | 6,144 |
| Attention heads | 64 (KV heads 8 → GQA) |
| **vocab_size** | **153,600** |
| Experts per token | 8 |
| Intermediate size | 18,432 (dense) / 2,048 (MoE) |
| **Max context** | **262,144** |
| Special: LLLG attention (Local-Local-Local-Global) | EXAONE 시리즈 고유 |

vocab_size 153,600은 [[qwen-25-family]] 72B(152,064)와 1,536 차이로 유사하나 special token 패턴이 완전히 다름.

## 검증 사용 모델 (LLM-DNA)

`trust_remote_code=True` 필수 (`exaone_moe` 커스텀 modeling). ml.p5.48xlarge fp16에 충분히 적재 (~470GB).

## 차별화 요소

- **256K context**: 본 프로젝트 검증 대상 중 최장. 별도 engineering 필요.
- **LLLG attention**: Local 3회 + Global 1회 패턴, EXAONE 4.0부터 도입.
- **Stanford AI Index 2026 등재**: 정정판에 K-EXAONE / EXAONE 4.0 (32B) / EXAONE PASS 2.0 / EXAONE Deep (32B) 모두 포함.

## 진화 가설

LG는 EXAONE 1.0 → 3.0 → 4.0 → K-EXAONE으로 5년+ 일관된 자체 시리즈를 유지. 이 시리즈 내 함수적 연속성이 [[neighbor-joining]] 트리에 자체 clade로 드러나야 정합적.

만약 K-EXAONE이 LG 자체 EXAONE 4.0 (32B)에서 확장된 것이라면 두 모델 간 거리는 매우 가까워야 하고, 이는 from scratch가 아니라 "**자체 모델의 scale-up + MoE 확장**"이라는 설명과 정합. 본 프로젝트 LLM-DNA 분석으로 이를 검증할 수 있다.

## LLLG Attention 의미

A.X-K1의 GQA(64-head MHA) vs K-EXAONE의 LLLG(blocky local + periodic global)는 함수적으로 다른 attention 패턴을 만든다. 이는 [[ax-k1]]과 K-EXAONE이 같은 한국 모델군 내에서도 거리가 있을 것이란 가설을 뒷받침.

## 관련 페이지

- [[dokpamo-project]] — 사업 맥락
- [[architecture-analysis]] — config.json 비교 방법론
- [[ax-k1]], [[solar-open-100b]] — 같은 Phase 2 진출 한국 모델
