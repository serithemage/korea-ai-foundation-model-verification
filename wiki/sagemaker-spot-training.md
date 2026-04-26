---
title: "SageMaker Spot Training (LLM-DNA용)"
type: policy
tags: [aws, sagemaker, spot, infrastructure]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# SageMaker Spot Training (LLM-DNA용)

본 프로젝트의 [[llm-dna-overview]] 추출은 GPU 추론 워크로드. SageMaker managed spot training이 **per-second billing + 무관리 + spot 70% 할인**으로 가장 비용 효율적.

본 프로젝트 인프라 코드: `experiments/llm-dna/cdk/`. 운영 노트: `experiments/llm-dna/README.md`.

## 핵심 의사결정

### 왜 학습 인프라를 추론에 쓰는가
- DNA 추출은 100 prompt × 모델 forward = 분~수십분 (학습 아님)
- 그러나 SageMaker training job은 컨테이너 + GPU + S3 통합이 가장 매끄러움
- "HUGI" 패턴 (Hurry Up and Get Idle) — burst 후 즉시 종료, idle cost 0

### Region 선택 (us-east-1)
사용자 지정 + spot placement score 9 (p5.48xlarge 기준). p4d/g6e는 score 3으로 낮음 → **모든 인스턴스를 p5 패밀리로 통일**.

### 인스턴스 매핑

| 모델 사이즈 | 인스턴스 | spot 시간당 (~) |
|------------|---------|----------------|
| 8B~72B reference | ml.p5.4xlarge (1×H100) | $7~10 |
| 100B~340B target | ml.p5.48xlarge (8×H100 640GB) | $30~40 |

### Quota 신청
- p5.4xlarge spot: 4 (이미 보유)
- p5.48xlarge spot: 1 (2026-04-26 신청 → **즉시 자동 승인**, 예외 케이스)

## CDK 스택 구성

`experiments/llm-dna/cdk/lib/llm-dna-stack.ts`. 핵심 리소스:

| 리소스 | 용도 |
|--------|------|
| S3 Bucket (`llm-dna-<acct>-<region>`) | HF cache + DNA 벡터 출력. cache prefix는 30일 만료 |
| Secrets Manager | HF token (env var 노출 회피) |
| ECR Repository | 사전 빌드된 llm-dna 컨테이너 (cold start 단축) |
| IAM Role | SageMaker 실행 역할. S3 RW + Secret read + ECR pull |

## 실행 패턴

```python
# scripts/submit_dna.py 의 골격
from sagemaker.pytorch import PyTorch
estimator = PyTorch(
    role=os.environ["SAGEMAKER_ROLE_ARN"],
    instance_count=1,
    instance_type="ml.p5.48xlarge",
    image_uri=f"{ecr}/llm-dna:latest",
    use_spot_instances=True,
    max_run=7200,         # 2h 추론 충분
    max_wait=10800,       # spot 회수 시 3h 대기
    hyperparameters={"model": "upstage/Solar-Open-100B"},
    environment={
        "HF_TOKEN_SECRET_ARN": hf_secret_arn,
        "S3_OUTPUT": f"s3://{bucket}/dna/",
    },
    disable_profiler=True,  # p5에서 필수
)
estimator.fit(wait=False)
```

## 흔한 함정

| 증상 | 원인 | 대응 |
|------|------|------|
| Job stuck "Starting" >5min | spot capacity 없음 | placement score 다른 region 시도 |
| `ValidationException: Profiler not supported` | g7e/p5에서 profiler 미지원 | `disable_profiler=True` |
| `ResourceLimitExceeded` | spot quota 0 | `aws service-quotas request-service-quota-increase` |
| CUDA kernel 오류 (Flash Attn) | architecture 미스매치 | `torch.cuda.get_device_capability()` fallback |

## 비용 추정 (본 프로젝트)

| 항목 | 추정 |
|------|------|
| Reference 7개 (p5.4xl) | ~$15 |
| GLM-5-FP8 + DeepSeek-V2.5 (p5.48xl) | ~$60 |
| Korean 3종 (p5.48xl) | ~$150 |
| S3 + ECR + 데이터 전송 | <$5 |
| **합계** | **~$230** |

## 관련 페이지

- [[aws-cdk-typescript]] — 스택 구현 패턴
- [[huggingface-hub-usage]] — HF token + cache 관리
- [[llm-dna-package]] — 컨테이너에 들어가는 패키지
