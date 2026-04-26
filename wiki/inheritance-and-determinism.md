---
title: "Inheritance와 Genetic Determinism"
type: concept
tags: [llm-dna, theory]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# Inheritance와 Genetic Determinism

[[llm-dna-overview]] 논문이 수학적으로 증명한 두 가지 성질. DNA 벡터가 모델 계통 추적에 의미 있게 작동하기 위한 핵심 전제다.

## Inheritance

부모 모델 P와 그 fine-tuned/distilled 자식 C가 있을 때, **DNA(P)와 DNA(C) 사이의 거리는 P와 무관한 임의 모델 X와의 거리보다 작아야 한다.** 즉 계통 관계가 거리 함수에 보존된다.

논문은 functional representation의 [[random-projection]] 변환이 bi-Lipschitz 사상이 되도록 구성하여 이 성질을 보장한다 — 입력 공간의 거리가 출력 공간에서 일정 비율 안에서 보존된다는 의미.

본 프로젝트에서의 함의: 만약 [[solar-open-100b]]가 정말로 GLM(Zhipu)에서 fine-tune됐다면, DNA 거리는 [[glm-family]]의 어느 모델보다 매우 가깝게 나와야 한다. 반대로 거리가 [[mixtral-8x7b]]나 [[llama-3-family]]와 비슷하다면 from scratch의 강한 정황 증거가 된다.

## Genetic Determinism

같은 모델에 같은 추출 설정(시드·prompt·차원)을 적용하면 항상 동일한 DNA 벡터가 나와야 한다. 부동소수점 비결정성 때문에 완전 동일은 어렵지만, 모델 간 거리에 비해 무시 가능한 수준이다.

이 성질이 깨지면 거리 비교 자체가 불가능해지므로 결정적이다. 실용적으로 보장하려면:

- `--random-seed 42` 고정
- 같은 `dataset`/`max_samples`/`dna_dim` 사용
- 양자화 사용 시 동일한 양자화 설정 (4-bit vs fp16 비교 금지)
- GPU 비결정성을 피하기 위해 가능하면 같은 인스턴스 타입에서 추출

## 실험 검증

305개 모델에서 알려진 부모-자식 관계가 트리에 정확히 복원되는지 검증함으로써 두 성질이 실용적으로도 성립함을 확인했다. 본 프로젝트는 이 보장에 의존하여 7~10개 reference 모델만으로도 [[ax-k1]] 등의 계통을 의미 있게 추정할 수 있다.

## 한계와 주의

- Inheritance는 **거리 보존**이지 **방향 회복**이 아니다 — DNA 거리가 가깝다고 어느 쪽이 부모인지는 알 수 없다 ([[neighbor-joining]] 출력은 unrooted tree).
- Out-of-distribution probe set에서는 보장이 약해질 수 있다 — `dataset="rand"`는 그래서 도메인 편향을 줄이는 안전한 기본값.
