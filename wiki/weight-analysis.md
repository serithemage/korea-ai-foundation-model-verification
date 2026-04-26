---
title: "Weight 분석"
type: concept
tags: [verification, methodology, weight, cosine-similarity]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Weight 분석

모델 가중치(weight tensor)를 직접 비교해서 fine-tuning 여부를 판별하는 방법. Tokenizer 분석보다 직접적이지만 architecture가 같아야만 가능하다는 한계가 있다.

본 프로젝트 상세 문서: [`docs/02-weight-analysis.md`](../docs/02-weight-analysis.md).

## 분석 항목

### Layer별 Cosine Similarity
가장 표준적인 방법.
```
cos(W_A, W_B) = (W_A · W_B) / (||W_A|| ||W_B||)
```
Fine-tuned 모델은 base와 초기 layer에서 >0.95 유사도 유지. From scratch는 random init 효과로 <0.1.

### Weight Tensor 해싱
SHA-256 등으로 fingerprint 생성. 완전 일치는 fine-tune 강한 증거 (LayerNorm은 init 편향으로 우연한 일치 가능 — [[layernorm-fingerprint-fallacy]]).

### PCA 분포 분석
모델들의 weight를 PCA로 2~3차원에 사영해 시각화. Fine-tuned 모델들은 base 근처에 cluster.

### Centered Cosine
**LayerNorm 같은 layer는 init 분포가 1.0으로 편향**돼 있어 raw cosine이 부풀려진다. 평균 오프셋을 빼고 cosine 계산하는 centered cosine이 표준 권장.

## 판정 기준

| 지표 | Fine-tuning | From Scratch |
|------|-------------|--------------|
| 평균 Layer Cosine Sim | >0.90 | <0.3 |
| Embedding Cosine Sim | >0.98 | <0.1 |
| LayerNorm Centered Cosine Sim | >0.7 | ≈0 |

## 한계 (본 프로젝트 핵심)

**Architecture가 다르면 직접 비교 불가**. 본 프로젝트 검증 대상 3종은 모두 reference 모델과 architecture가 다름:

- [[solar-open-100b]]: SolarOpen, MoE 128+1
- [[k-exaone-236b]]: ExaoneMoE, MoE 8 active, LLLG attention
- [[ax-k1]]: AXK1 custom_code, MoE 192+1, MHA

이는 **weight 비교를 시도조차 할 수 없음** = **그 자체가 from scratch 정황 증거**라고 본 프로젝트 README는 해석.

## LLM-DNA가 메우는 빈자리

[[llm-dna-overview]]가 weight 분석의 한계를 우회한다. functional behavior(출력)를 비교하므로:
- Tokenizer 달라도 OK
- Architecture 달라도 OK (custom_code도)
- 단 인 출력 차원·임베딩 추출 방식만 통일하면 됨

본 프로젝트가 LLM-DNA를 도입한 핵심 이유 — weight 분석이 막힌 곳에서 functional 분석으로 돌파.

## EleutherAI의 LayerNorm 검증 사례

Solar-Open-100B 96.8% LayerNorm 유사도 주장에 대해 hyunwoongko가 [solar-vs-glm-vs-phi](https://github.com/hyunwoongko/solar-vs-glm-vs-phi)에서 반박. 같은 모델 다른 layer 사이에도 LayerNorm cosine이 0.99 수준 — init 편향의 결과지 진짜 유사가 아님. centered cosine으로 바로 잡으면 모델 간 유사도가 거의 0으로 떨어짐. → [[layernorm-fingerprint-fallacy]] 참조.

## 관련 페이지

- [[tokenizer-analysis]] — 보다 보편적
- [[architecture-analysis]] — config 수준
- [[layernorm-fingerprint-fallacy]] — weight 분석의 흔한 함정
- [[llm-dna-overview]] — weight 비교 불가 케이스의 대안
