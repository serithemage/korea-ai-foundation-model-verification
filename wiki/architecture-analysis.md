---
title: "Architecture 분석"
type: concept
tags: [verification, methodology, architecture, config]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Architecture 분석

`config.json`의 hyperparameter를 비교해서 두 모델이 같은 구조에서 출발했는지 확인. Fine-tuned 모델은 **base와 정확히 같은 architecture**여야 weight를 그대로 가져올 수 있으므로, config 일치는 fine-tune의 강력한 증거.

본 프로젝트 상세 문서: [`docs/03-architecture-analysis.md`](../docs/03-architecture-analysis.md).

## 비교 핵심 5개 항목

| 항목 | 의미 |
|------|------|
| `hidden_size` | embedding/layer 차원 |
| `num_hidden_layers` | transformer block 수 |
| `num_attention_heads` | MHA head 수 |
| `num_key_value_heads` | GQA의 KV head (head와 같으면 MHA) |
| `vocab_size` | embedding matrix 차원 결정 |

## MoE 추가 항목

| 항목 | 의미 |
|------|------|
| `n_routed_experts` (또는 `num_local_experts`) | routed expert 수 |
| `n_shared_experts` | shared expert 수 |
| `num_experts_per_tok` | 토큰당 활성화 expert 수 |
| `moe_intermediate_size` | MoE FFN 내부 차원 |

## RoPE 설정

| 항목 | 의미 |
|------|------|
| `rope_theta` | RoPE base frequency. Llama 3.x 500K, Qwen 2.5 1M, Solar 1M, A.X-K1 10K (특이) |
| `rope_scaling` | YaRN/Linear 등 long context 보조 |
| `max_position_embeddings` | 컨텍스트 길이 |

## 본 프로젝트 비교표

| 모델 | hidden | layers | heads | KV | vocab | RoPE θ | max_ctx | experts |
|------|--------|--------|-------|-----|-------|--------|---------|--------|
| [[solar-open-100b]] | 4096 | 48 | 64 | 8 | 196,608 | 1M | 131,072 | 128+1 |
| [[k-exaone-236b]] | 6144 | 48 | 64 | 8 | 153,600 | n/a | 262,144 | n+1 (a23B) |
| [[ax-k1]] | 7168 | 61 | 64 | **64 (MHA)** | 163,840 | **10,000** | 131,072 | 192+1 |
| [[llama-3-family]] 70B | 8192 | 80 | 64 | 8 | 128,256 | 500K | 131,072 | dense |
| [[qwen-25-family]] 72B | 8192 | 80 | 64 | 8 | 152,064 | 1M | 131,072 | dense |

핵심 5개 + MoE 항목 모두 reference와 0개 완전 일치 → 독립 설계 (from scratch 지지).

## 판정 로직

| 상황 | 해석 |
|------|------|
| 핵심 5개 모두 어떤 reference와 정확 일치 | Fine-tuning 강한 의심 |
| 1~3개만 일치 | 비슷한 디자인 영감, weight 차용은 별도 검증 |
| 0개 일치 | 독립 설계 (from scratch 지지) |

## 한계

- Config 일치만으로 weight 차용 단정 못 함 (단순히 같은 디자인 채택일 수 있음)
- 학술 논문 ([[model-provenance-testing]] arXiv:2502.00706)이 지적: "Fine-tuning은 architecture 보존하므로 동일 dim 모델이 파생작일 수 있음"
- Custom architecture (예: [[ax-k1]] AXK1)는 표준 모델과 매핑이 어려워 비교 자체가 까다로움

## LLM-DNA와의 관계

Architecture 분석은 정적·구조적, [[llm-dna-overview]]는 동적·기능적. 두 결과가 일치하면 결론 강화. 불일치하면 추가 조사 필요:
- Architecture 다른데 LLM-DNA 가까움 → 같은 BPE/데이터셋으로 다른 architecture 학습한 케이스
- Architecture 같은데 LLM-DNA 멀음 → fine-tune 의심을 약화 (functional하게 다르게 진화)

## 관련 페이지

- [[tokenizer-analysis]] — 가장 보편적 1차 지표
- [[weight-analysis]] — architecture 같을 때만 유효
- [[behavior-analysis]] — 행동 기반 검증
