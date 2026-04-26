---
title: "DNA 추출 파이프라인"
type: concept
tags: [llm-dna, algorithm]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: [raw/2026-04-26_phylogenetic-methods.md]
---

# DNA 추출 파이프라인

[[llm-dna-overview]]의 핵심 알고리즘. 임의 LLM을 입력으로 받아 고정 차원(기본 128)의 DNA 벡터를 출력한다.

## 4단계

### 1. Probe set 구축
다양한 도메인을 커버하는 입력 프롬프트 셋을 준비한다. [[llm-dna-package]] 기본값은 `dataset="rand"`로, 시드 기반 무작위 토큰 시퀀스를 생성한다. 이 방식은 데이터셋 다운로드가 필요 없고, 시드만 고정하면 모델 간 비교가 공정해진다. `max_samples`는 기본 100.

### 2. Functional output 수집
각 prompt를 모델에 입력하고 출력 임베딩(혹은 hidden state)을 모은다. `embedding_merge` 옵션으로 토큰별 임베딩을 어떻게 합칠지 선택한다 — `mean`(평균, 기본), `sum`, `max`, `concat`. Decoder-only 모델은 마지막 hidden state, encoder-decoder는 encoder 출력을 사용한다.

### 3. Random projection으로 차원 축소
[[random-projection]]을 적용해 고차원 functional matrix를 `dna_dim`(기본 128) 차원으로 압축한다. 대안으로 `pca`, `svd`도 선택 가능하지만, random projection이 inheritance 성질 보존에 가장 안정적이라고 논문은 보고한다.

### 4. DNA 벡터 출력
결과는 numpy 배열로 저장된다 (KB 단위). 모델 자체의 weight(수백 GB)와 비교해 압축률이 극단적이며, 이후 거리 계산·트리 구축이 로컬 환경에서 가능해진다.

## 결정성

같은 모델에 같은 시드를 주면 거의 동일한 DNA 벡터가 나온다. 부동소수점 비결정성 때문에 완전히 같지는 않지만, 서로 다른 모델 간 거리에 비해 무시 가능한 수준이다. 이 결정성이 [[phylogenetic-tree]] 구축의 안정성을 보장한다.

## 비용

DNA 추출은 추론 워크로드 (학습이 아님). 100 prompt × 모델 forward pass = 모델 크기에 따라 분~수십분. 100B+ MoE 모델도 [[sagemaker-spot-training]] p5.48xlarge 한 대에 1~2시간이면 충분하다.

## 코드 진입점

```python
from llm_dna import DNAExtractionConfig, calc_dna
config = DNAExtractionConfig(
    model_name="upstage/Solar-Open-100B",
    dataset="rand",
    max_samples=100,
    dna_dim=128,
    reduction_method="random_projection",
    random_seed=42,
    trust_remote_code=True,
)
result = calc_dna(config)  # result.vector: (128,) numpy array
```

CLI: `calc-dna --model-name <id> --dataset rand --max-samples 100 --dna-dim 128`
