---
title: "DeepSeek-V2.5"
type: entity
tags: [reference-model, deepseek, moe]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# DeepSeek-V2.5

DeepSeek AI의 236B MoE 모델. 본 프로젝트에서 **heavy MoE reference**로 사용. HuggingFace: [deepseek-ai/DeepSeek-V2.5](https://huggingface.co/deepseek-ai/DeepSeek-V2.5). 2024-09-05 공개.

## 핵심 사양

| 항목 | 값 |
|------|----|
| Architecture | DeepseekV2ForCausalLM (`custom_code`) |
| 총 파라미터 | 236B |
| 활성 파라미터 | 21B (per token) |
| arXiv | [2405.04434](https://arxiv.org/abs/2405.04434) |
| 인스턴스 | ml.p5.48xlarge fp16 |

원래 사용자가 DeepSeek-V3 (671B)를 reference 후보로 제시했으나, 8×H100 1대에 맞지 않아(671B fp16 = 1.3TB) 같은 패밀리의 V2.5(236B)로 대체. 모델 패밀리 시그널은 동일하게 잡힌다.

## Reference로 선택한 이유

[[k-exaone-236b]]가 정확히 같은 236B 총 파라미터, 23B vs 21B 활성 파라미터로 매우 비슷한 스케일. 이 두 모델이 DNA 거리에서 서로 가까운지 vs 멀리 떨어지는지가 흥미로운 비교점이 된다.

만약 DeepSeek와 K-EXAONE이 가까우면 "비슷한 스케일·구조의 MoE는 비슷하게 행동"이라는 가설 강화. 만약 멀면 "K-EXAONE은 EXAONE 4.0 시리즈의 자체 진화"라는 LG 주장과 정합.

## DeepSeek MoE 특이점

- **DeepSeekMoE**: shared experts + routed experts 조합. [[solar-open-100b]] (128+1) / [[ax-k1]] (192+1) / [[k-exaone-236b]]가 모두 이 패턴 채택.
- **MLA (Multi-Latent Attention)**: KV cache 압축 기법. 한국 모델은 채택하지 않음.
- DeepSeek 시리즈는 중국에서 가장 영향력 큰 오픈 MoE. 한국 모델 설계자들이 참고했을 가능성 높음.

## LLM-DNA 분석 의의

DeepSeek-V2.5와 [[glm-family]] (둘 다 중국 origin)이 가까운지, 아니면 dense·MoE 차이가 더 큰지 확인 가능. 또 DeepSeek vs Mixtral 거리가 두 MoE 모델 간 거리의 baseline이 된다.

## 함정

`custom_code: trust_remote_code=True` 필수. DeepSeek 시리즈는 빈번한 transformers 호환성 변경이 있으므로 본 프로젝트 실행 시점 transformers 버전과 호환성 확인 필요.
