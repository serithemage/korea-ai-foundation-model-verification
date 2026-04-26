---
title: "Tokenizer 분석"
type: concept
tags: [verification, methodology, tokenizer, bpe]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Tokenizer 분석

LLM의 from scratch 학습 여부를 검증하는 가장 접근성 높은 방법. Tokenizer는 학습 비용이 매우 크기 때문에 fine-tuning 시에는 거의 그대로 재사용된다 — 따라서 tokenizer 유사도가 모델 기원의 강력한 지표가 된다.

본 프로젝트 상세 문서: [`docs/01-tokenizer-analysis.md`](../docs/01-tokenizer-analysis.md).

## 분석 항목

### 1. vocab_size 비교
가장 직관적. 정확히 일치하면 동일 tokenizer 사용 강력 의심. 본 프로젝트 예:

| 모델 | vocab_size | 비교 결과 |
|------|-----------|----------|
| [[solar-open-100b]] | 196,608 | 모든 reference와 불일치 → from scratch 강력 지지 |
| [[k-exaone-236b]] | 153,600 | Qwen 152,064와 1,536 차이 → 의심 가치 |
| [[ax-k1]] | 163,840 | 모든 reference와 불일치 → from scratch 강력 지지 |

### 2. Token 중복률
두 tokenizer의 vocabulary 집합 교집합 / 합집합. 90% 이상이면 같은 BPE 코퍼스 의심.

### 3. BPE Merge Rules 순서
가장 강력한 fingerprint. tokenizer.json의 merges 배열 처음 N개를 비교. 동일한 BPE 학습 과정을 거쳤다면 같은 순서로 토큰이 병합된다. 본 프로젝트의 HyperCLOVAX-SEED 분석에서 Qwen2.5와 처음 20개 merge가 완전 일치 — 강한 의심 증거.

### 4. Special tokens 패턴
- Llama 2 / SentencePiece: `<s>`, `</s>`, `<pad>`
- Llama 3: `<|begin_of_text|>`, `<|eot_id|>`
- Qwen ChatML: `<|im_start|>`, `<|im_end|>`
- ChatGLM: `<|user|>`, `<|assistant|>`

스타일 일치는 같은 family origin 강력 시사.

## 판정 기준

| 중복률 | 해석 |
|--------|------|
| >98% | Fine-tuning 가능성 높음 |
| 90~98% | Continued pre-training 또는 vocab 확장 |
| <90% | From scratch 학습 강력 증거 |

## 한계

- Tokenizer 일치만으로 weight까지 차용했다고 단정 못 함 (단순히 같은 tokenizer를 채택했을 수도)
- BPE 학습이 동일 코퍼스 + 동일 설정이면 우연히 비슷한 merge 순서 가능
- vocab_size 1,536 정도 차이는 special token만 추가/제거한 결과일 수 있음

## LLM-DNA와의 관계

[[llm-dna-overview]]는 functional behavior를 보지 tokenizer는 직접 보지 않는다. 두 분석은 **상보적**:
- Tokenizer 분석: vocabulary·BPE 수준의 정적 비교
- LLM-DNA: 출력 행동의 functional 비교

만약 Tokenizer가 같은 family인데 LLM-DNA에서 거리가 멀다면 "tokenizer만 차용, weight는 from scratch" 가능성. 반대로 둘 다 가깝다면 fine-tune 의심 강화.

## 관련 페이지

- [[weight-analysis]] — weight 직접 비교
- [[architecture-analysis]] — config 비교
- [[layernorm-fingerprint-fallacy]] — 통계량 비교의 함정
