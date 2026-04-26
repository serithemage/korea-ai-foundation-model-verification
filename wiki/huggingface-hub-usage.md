---
title: "HuggingFace Hub 사용 패턴"
type: concept
tags: [huggingface, model-download, caching]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# HuggingFace Hub 사용 패턴

본 프로젝트는 ~10개 모델(총 2TB+)을 HF에서 다운로드. 효율적 cache + auth + gated 모델 처리가 비용·시간에 직결.

## Token 환경

| 환경 | 변수 |
|------|------|
| 로컬 (~/.zshrc) | `HUGGINGFACE_ACCESS_TOKEN` |
| HF SDK 자동 인식 | `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` |
| 본 프로젝트 클라우드 | Secrets Manager `llm-dna/hf-token` |

llm-dna CLI는 `--token <value>` 또는 `HF_TOKEN` env. 명시적 권장:
```bash
export HF_TOKEN=$HUGGINGFACE_ACCESS_TOKEN
calc-dna --model-name <id>
```

## Cache 전략

### 로컬
```bash
export HF_HOME=~/.cache/huggingface  # 기본
# 또는 프로젝트 격리:
export HF_HOME=/path/to/experiments/llm-dna/hf_cache
```

### SageMaker
컨테이너 내부 `/tmp`는 휘발성. Spot 회수 시 다시 다운로드. 대안:
- **S3 Mount (FSx for Lustre)**: 한 번 다운로드 → 모든 job에서 공유
- **사전 baking**: 컨테이너 빌드 시 작은 모델은 미리 받아서 이미지에 포함
- **EFS**: 작은 모델용 (대형은 비싸짐)

본 프로젝트는 `s3://<bucket>/cache/hf/`에 동기화하는 패턴 권장 (lifecycle 30일 만료로 비용 통제).

## Gated 모델 (Llama 등)

Meta-Llama 시리즈는 license accept 후에만 다운로드 가능. 프로세스:
1. HF 웹사이트에서 모델 페이지 → "Request access"
2. Meta 폼 작성 (즉시 승인)
3. 같은 token으로 자동 다운로드

본 프로젝트의 [[llama-3-family]]는 이 단계 사전 완료 필요.

## API 메타데이터 fetch

다운로드 없이 config만 필요할 때:
```bash
curl -sL "https://huggingface.co/api/models/upstage/Solar-Open-100B" | jq '.config'
curl -sL "https://huggingface.co/upstage/Solar-Open-100B/resolve/main/config.json"
```

본 프로젝트의 [[architecture-analysis]] 비교는 이 방식으로 모든 모델 config를 다운로드 없이 수집했다.

## Custom code (`trust_remote_code=True`)

[[ax-k1]] (`AXK1ForCausalLM`), [[k-exaone-236b]] (`exaone_moe`), [[deepseek-v25]] (DeepSeek 시리즈)는 모두 `trust_remote_code=True` 필요. **보안 주의**: HF에서 받은 Python 코드를 신뢰하고 실행한다는 의미. 격리된 환경(SageMaker 컨테이너)에서만 실행 권장.

## Rate limit

미인증 요청은 분당 ~5 req로 제한. **HF_TOKEN 설정 필수**. 본 프로젝트는 distilgpt2 sanity check 시에도 token 미설정 시 경고 표시:
```
Warning: You are sending unauthenticated requests to the HF Hub.
Please set a HF_TOKEN to enable higher rate limits and faster downloads.
```

## 다운로드 가속

- `huggingface_hub.snapshot_download(..., max_workers=8)` 병렬 다운로드
- `HF_HUB_ENABLE_HF_TRANSFER=1` + `pip install hf_transfer` (Rust binary, 큰 모델에 효과적)
- AWS region us-east-1은 HF CDN과 가까워 빠름

본 프로젝트는 `hf_transfer` 활성화 권장 (519B [[ax-k1]] 다운로드 시간 결정적).

## 관련 페이지

- [[sagemaker-spot-training]] — cache 위치
- [[aws-cdk-typescript]] — Secrets Manager로 token 관리
