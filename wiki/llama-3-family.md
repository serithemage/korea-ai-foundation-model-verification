---
title: "Llama 3 패밀리"
type: entity
tags: [reference-model, meta, dense]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Llama 3 패밀리

Meta가 공개한 dense decoder-only LLM 시리즈. 본 프로젝트 [[llm-dna-overview]] 분석에서 가장 보편적인 baseline reference로 사용.

## 본 프로젝트 사용 변종

| 모델 | 파라미터 | vocab_size | Architecture | 인스턴스 |
|------|---------|-----------|-------------|---------|
| meta-llama/Llama-3.1-8B | 8B | 128,256 | LlamaForCausalLM | ml.p5.4xlarge fp16 |
| meta-llama/Llama-3.3-70B-Instruct | 70B | 128,256 | LlamaForCausalLM | ml.p5.4xlarge 4-bit |

두 모델 모두 GQA (Llama 3.x 기본), RoPE θ=500,000, max_position_embeddings=131,072.

## Reference로 선택한 이유

- **개방성**: HF 직접 다운로드 가능. License accept만 필요.
- **광범위한 영향**: 한국 모델 일부가 Llama tokenizer를 미세 변형해 사용한 의혹이 있음 ([[k-exaone-236b]] vocab_size 153,600 vs Llama 128,256 — 차이는 있지만 same-family bpe 가능성). NJ 트리에서 한국 모델이 Llama clade와 거리가 먼지를 가시화.
- **사이즈 다양성**: 8B와 70B를 함께 넣으면 같은 family 내 사이즈만 다른 케이스의 distance를 calibration할 수 있음.

## Tokenizer 핵심

`<|begin_of_text|>`, `<|eot_id|>` 등 새로운 special token 패턴 도입 (Llama 2의 SentencePiece-style `<s>`, `</s>`와 다름). 이 special token style이 한국 모델에 그대로 등장하면 fine-tune 의심 증거.

## LLM-DNA 분석 의의

[[llama-3-family]]와 한국 모델([[solar-open-100b]], [[k-exaone-236b]], [[ax-k1]]) 사이의 cosine 거리가 충분히 크면 "Llama 기반 fine-tune이 아님"의 정황 증거. 반대로 어느 한국 모델이 Llama-3.3-70B 근처에 cluster되면 추가 검증 필요.

## 함정

Llama 3.3-70B-**Instruct**는 base가 아니라 instruction-tuned. 이는 RLHF/DPO를 거친 모델로 base와 functional behavior가 다를 수 있음. 가능하면 base 모델(Llama-3.1-70B)도 함께 추출해 비교하는 것이 이상적이나 디스크·비용 부담으로 본 프로젝트는 Instruct만 사용.
