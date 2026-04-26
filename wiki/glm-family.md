---
title: "GLM 패밀리 (Zhipu)"
type: entity
tags: [reference-model, zhipu, chatglm, controversy]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# GLM 패밀리 (Zhipu)

Zhipu AI(智谱AI)가 공개한 GLM(General Language Model) 시리즈. **본 프로젝트의 핵심 비교 대상** — [[solar-open-100b]] 코드 차용 의혹의 직접 당사자.

## 본 프로젝트 사용 변종

| 모델 | 파라미터 | Architecture | 인스턴스 |
|------|---------|-------------|---------|
| zai-org/glm-4-9b-chat-hf | 9B | GlmForCausalLM | ml.p5.4xlarge fp16 |
| zai-org/GLM-5-FP8 | ~340B FP8 native | GlmMoeDsaForCausalLM | ml.p5.48xlarge fp16 |

GLM-5는 MoE + DSA(Deep Self-Attention?) 구조의 새 architecture. FP8 native로 학습돼 4-bit 양자화 없이도 적재 효율적.

## 핵심 맥락

2026-01-01 Solar-Open-100B 공개 직후 PsionicAI 고석현 CEO가 "Solar 학습 코드에 GLM 저작권 표기가 남아 있다", "LayerNorm 유사도가 GLM과 96.8%"라며 차용 의혹 제기. 이후 [[layernorm-fingerprint-fallacy]] 분석으로 LayerNorm 비교의 방법론적 결함이 드러나면서 의혹은 약화됐지만, "정확한 검증"은 미완 상태.

## Reference로 선택한 이유

- 직접 비교의 필요성: 의혹의 당사자를 트리에 함께 넣어야 결과가 의미 있음.
- GLM-4 (9B)와 GLM-5 (340B 클래스) 두 사이즈를 같이 넣어 GLM family 내부 distance를 calibration.

만약 [[solar-open-100b]]가 정말 GLM 기반이라면 GLM-5-FP8과 가까운 위치에 cluster되어야 한다.

## Tokenizer 핵심

GLM-4까지는 ChatGLM tokenizer (sentencepiece-based, 중·영 강세). GLM-5는 새 tokenizer로 vocab 확장.

## LLM-DNA 분석 의의

이 페이지의 가장 중요한 결과 — **Solar-GLM 거리 vs Solar-Llama 거리**. 만약 Solar가:
- GLM clade에 가까움 → fine-tune 가설 강화 (단 weight 직접 검증 권장)
- 그 외 reference 어느 곳과도 멀고 [[k-exaone-236b]]/[[ax-k1]]과 한국 군 형성 → from scratch 강한 증거
- Llama 또는 Mixtral 근처 → 또 다른 base 가설 검토

## 함정

GLM-5-FP8은 비교적 신모델(2026-02-11 공개). DNA 추출 시 `trust_remote_code=True`가 필요할 가능성 높음. 사전 sanity로 작은 변종(GLM-4 9B)을 먼저 돌려 파이프라인 안정성 확인 권장.
