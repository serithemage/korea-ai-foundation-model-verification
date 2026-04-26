---
title: "SKT A.X-K1"
type: entity
tags: [korean-llm, moe, dokpamo, skt]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# SKT A.X-K1

[[dokpamo-project]] 1차 평가 2위로 Phase 2 진출. **519B 파라미터로 한국 최초 ultra-large LLM**. HuggingFace: [skt/A.X-K1](https://huggingface.co/skt/A.X-K1). 2025-12-29 공개.

## 핵심 사양

| 항목 | 값 |
|------|----|
| Architecture | `AXK1ForCausalLM` (`AXK1` model_type, **custom_code**) |
| 총 파라미터 | 519B |
| MoE | 192 routed + 1 shared experts |
| Layers | **61** (검증 대상 중 최다) |
| Hidden size | 7,168 (검증 대상 중 최대) |
| Attention heads | 64 (**KV heads 64 → MHA, not GQA**) |
| **vocab_size** | **163,840** |
| Experts per token | 8 |
| Intermediate size | 18,432 (dense) / 2,048 (MoE) |
| RoPE θ | **10,000** ([[solar-open-100b]] 1M 대비 100배 작음) |
| Max context | 131,072 |
| dtype | bfloat16 |

## 검증 사용 모델 (LLM-DNA)

**`trust_remote_code=True` 필수, 그리고 본 프로젝트의 가장 도전적인 케이스.** 519B fp16 = ~1TB → ml.p5.48xlarge (8×H100 = 640GB)에 fp16으로는 안 들어간다. 8-bit 양자화(약 520GB)도 빠듯. 4-bit(약 260GB)로 추론하거나, 멀티노드 검토 필요.

* `custom_code` 호출 자체가 첫 시도라 sanity가 어렵다 — Phase 0에서 모델 로드만이라도 작은 인스턴스에서 검증하는 것이 안전.

## 차별화 요소

- **193 experts**: 검증 대상 중 가장 많은 expert. 독자적 설계 의도가 명확.
- **MHA 유지**: 다른 한국 모델은 GQA로 가는 추세인데 A.X-K1은 KV heads 64로 MHA 유지. KV cache 크기가 크지만 성능을 우선한 선택.
- **rope_theta 10,000**: 작은 값. [[k-exaone-236b]] / [[solar-open-100b]]가 모두 1M+ 사용하는 것과 대조. RoPE scaling을 별도로 적용한 흔적 가능.

## SKT의 풀스택 AI 전략

A.X-K1은 SKT의 "Full-Stack AI" 비전(인프라 + 모델 + 서비스)의 핵심으로 포지션. 2026-04-22~24 World IT Show 2026에서 공개 시연.

## LLM-DNA 분석 의의

custom_code 모델은 [[architecture-analysis]] config 비교가 까다롭다 — 표준 transformers 클래스와 mapping이 안 됨. [[llm-dna-overview]] functional 접근법이 이런 케이스에 특히 빛난다 — 내부가 어떻든 "출력 행동"만 보면 되므로.

만약 A.X-K1이 진짜 from scratch라면 reference 어느 군과도 멀리 떨어진 위치에 자체 clade를 형성. 같은 한국 모델 [[k-exaone-236b]] / [[solar-open-100b]]와의 거리가 어떻게 나오는지가 흥미로운 관전 포인트.

## 관련 페이지

- [[dokpamo-project]] — 사업 맥락
- [[architecture-analysis]] — config 비교 방법론
- [[sagemaker-spot-training]] — 인프라 (이 모델이 가장 빡빡)
- [[k-exaone-236b]], [[solar-open-100b]] — 같은 Phase 2 진출 한국 모델
