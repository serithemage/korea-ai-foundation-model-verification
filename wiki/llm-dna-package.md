---
title: "llm-dna Python 패키지"
type: entity
tags: [llm-dna, software, python]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# llm-dna Python 패키지

[Xtra-Computing/LLM-DNA](https://github.com/Xtra-Computing/LLM-DNA) 저장소의 공식 구현. PyPI에 [llm-dna](https://pypi.org/project/llm-dna/)로 등록.

## 설치

```bash
pip install llm-dna                  # 기본
pip install "llm-dna[apple]"         # MLX 백엔드 (Apple Silicon)
pip install "llm-dna[quantization]"  # bitsandbytes·GPTQ·compressed-tensors
pip install "llm-dna[model_families]" # mamba-ssm·timm
pip install "llm-dna[full]"          # 위 모두
```

본 프로젝트는 `experiments/llm-dna/.venv`에서 `[apple]`로 설치 (Mac 로컬 sanity check용).

## 핵심 API

```python
from llm_dna import DNAExtractionConfig, calc_dna
result = calc_dna(DNAExtractionConfig(
    model_name="distilgpt2",
    dataset="rand",
    max_samples=100,
    dna_dim=128,
    reduction_method="random_projection",
    trust_remote_code=True,  # AXK1, EXAONE 등 custom_code 모델에 필수
))
```

`result.vector`는 `(dna_dim,)` 형상의 numpy 배열. `result.output_path`로 디스크 위치 확인.

## CLI

```bash
calc-dna --model-name <id> --dataset rand --max-samples 100 --dna-dim 128 \
         --reduction-method random_projection --random-seed 42 \
         --trust-remote-code --device cuda \
         --output-dir ./out
```

배치 실행은 `--llm-list configs/llm_list.txt`로 round-robin GPU 할당. `--continue-on-error`로 한 모델 실패가 전체를 막지 않게 설정.

## 지원 백엔드

| 환경 | device 옵션 |
|------|------------|
| NVIDIA GPU | `cuda`, `cuda:0` |
| CPU | `cpu` |
| Apple MPS | **미지원** (cpu fallback) |

API 모델([[deepseek-v25]] DeepSeek API 등)은 `model_type=openrouter|openai|gemini|anthropic`로 호출 가능 — 다운로드 없이 쿼리 기반으로 DNA 추출. 단, 본 프로젝트 대상([[solar-open-100b]], [[k-exaone-236b]], [[ax-k1]])은 OpenRouter에 없으므로 weight 다운로드 + [[sagemaker-spot-training]] 필요.

## 출력 구조

```
out/
└── <dataset>/                # 예: rand/
    └── <model-id>/           # 예: distilgpt2/
        ├── responses.json    # 모델 forward 결과
        └── dna_vector.npy    # 최종 DNA 벡터
```

여러 모델 비교 시 `out/rand/*/dna_vector.npy`를 모아 거리 행렬을 만들고 [[neighbor-joining]]에 입력한다.
