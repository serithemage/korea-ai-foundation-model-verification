---
title: "Qwen 2.5 패밀리"
type: entity
tags: [reference-model, alibaba, dense, korean-relevant]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Qwen 2.5 패밀리

Alibaba(Qwen 팀)가 공개한 dense LLM 시리즈. 본 프로젝트에서 **가장 비중 있는 reference** — Naver HyperCLOVAX-SEED와 SKT [[ax-k1]]의 일부 의혹과 직접 관련.

## 본 프로젝트 사용 변종

| 모델 | 파라미터 | vocab_size | Architecture | 인스턴스 |
|------|---------|-----------|-------------|---------|
| Qwen/Qwen2.5-7B | 7B | 152,064 | Qwen2ForCausalLM | ml.p5.4xlarge fp16 |
| Qwen/Qwen2.5-32B | 32B | 152,064 | Qwen2ForCausalLM | ml.p5.4xlarge fp16 |
| Qwen/Qwen2.5-72B | 72B | 152,064 | Qwen2ForCausalLM | ml.p5.4xlarge 4-bit |

GQA, RoPE θ=1,000,000, 멀티링구얼 (특히 한·중·일 강세). vocab_size 152,064는 [[k-exaone-236b]]의 153,600과 1,536 차이 — special token만 다른 가능성 시사.

## 한국 모델과의 의혹

- **HyperCLOVAX-SEED tokenizer**: BPE merge rules 처음 20개가 Qwen2.5와 완전 일치. 본 프로젝트 tutorial Q11 분석. ([[from-scratch-debate]] 참조)
- **HyperCLOVAX-SEED Vision Encoder**: Qwen2.5 ViT 파인튜닝 사용 (네이버 공식 인정).
- **K-EXAONE vocab_size**: Qwen 152K와 1,536 차이. 우연인지 같은 BPE 학습 코퍼스인지 미확정.

## Reference로 선택한 이유

위 의혹들 때문에 한국 모델이 [[neighbor-joining]] 트리에서 Qwen clade에 가까이 가는지가 핵심 관전 포인트. 사이즈 3개를 모두 넣어 같은 family 내 calibration도 가능.

## Tokenizer 핵심

Special token: `<|im_start|>`, `<|im_end|>` 등 ChatML-style. 한·중 다국어 토큰 풍부. SentencePiece-based.

## LLM-DNA 분석 의의

만약 [[ax-k1]] 또는 [[k-exaone-236b]]의 DNA 벡터가 Qwen 7B/32B/72B와 일관되게 가까우면 "동일 BPE 코퍼스 + 비슷한 학습 분포"의 강한 정황 증거. 단 functional similarity가 곧 weight 차용은 아니라는 점 ([[layernorm-fingerprint-fallacy]]) 유의.

## 함정

Qwen 2.5-32B / 72B는 license가 "qwen" 라이선스로 일부 제한 있음. 본 프로젝트는 inference만 사용 (재배포·재학습 X)이라 무관.
