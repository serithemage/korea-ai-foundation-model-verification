---
title: "Model Provenance Testing (학술 연구)"
type: concept
tags: [verification, methodology, academic, arxiv]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Model Provenance Testing (학술 연구)

LLM 기원 검증에 대한 학술 연구. 본 프로젝트의 방법론적 backbone. 핵심 참고: [arXiv:2502.00706](https://arxiv.org/abs/2502.00706) — *Model Provenance Testing for Large Language Models*.

## 핵심 메시지

| 방법 | 정확도 | 비고 |
|------|--------|------|
| **Black-box Output Similarity** | 90~95% precision | 동일 prompt 출력 토큰 일치율 통계 |
| Config 비교 | 불충분 | Fine-tuning은 architecture 보존하므로 동일 dim ≠ 파생 X |
| Weight 비교 | 유효 | White-box (weight 다운로드) 필요 |

가장 학술적으로 신뢰 받는 방법은 **black-box output similarity** — 두 모델에 같은 prompt를 주고 출력을 통계적으로 비교. [[llm-dna-overview]]도 본질적으로 이 접근의 정교한 형태.

## 본 프로젝트 한계 인정

본 프로젝트의 4가지 분석법 ([[tokenizer-analysis]], [[weight-analysis]], [[architecture-analysis]], [[behavior-analysis]])은 학술 기준에 비하면 모두 약점 있음:

- **vocab_size 비교**: Fine-tuning 시 tokenizer 그대로 사용 → vocab_size 일치가 파생 증거가 됨. 본 프로젝트는 "vocab_size **불일치** = from scratch 지지"로 사용 (반대 방향). 이는 valid한 논리지만 충분조건이 아닌 강한 정황.
- **Config 비교**: 위 학술 연구가 명시적으로 비판한 방법.

## 본 프로젝트가 LLM-DNA를 도입한 정당성

[[llm-dna-overview]]는 black-box output similarity의 정량적·확장 가능한 구현. 학술적으로 가장 신뢰받는 접근을 본 프로젝트가 적용함으로써 검증 신뢰도를 한 단계 끌어올린다.

## Yi-Llama 사례

01.AI Yi-34B에 대한 파생 의혹과 [EleutherAI 분석](https://blog.eleuther.ai/nyt-yi-34b-response/)이 시사하는 것:

- **Architecture 동일성만으로는 파생 증거 불충분**
- Yi는 결과적으로 독립 학습으로 결론
- 정밀 검증에는 multiple methodology cross-check 필수

본 프로젝트도 단일 분석에 의존하지 않고 [[tokenizer-analysis]] + [[architecture-analysis]] + [[llm-dna-overview]] cross-validation으로 신뢰도 확보.

## 추천 학술 자료

- arXiv:2502.00706 (Model Provenance Testing)
- ICLR'26 [[llm-dna-overview]] 논문
- EleutherAI [Yi-Llama 분석](https://blog.eleuther.ai/nyt-yi-34b-response/)
- Magazine Sebastian Raschka, [LLM Evaluation Approaches](https://magazine.sebastianraschka.com/p/llm-evaluation-4-approaches)

## 관련 페이지

- [[llm-dna-overview]] — 학술적 기반
- [[layernorm-fingerprint-fallacy]] — 잘못된 분석의 사례
- [[from-scratch-debate]] — 정의의 모호성
