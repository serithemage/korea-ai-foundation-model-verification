# LLM-DNA 혈통 분석 실험

본 실험은 [LLM-DNA](https://github.com/Xtra-Computing/LLM-DNA) (ICLR'26 Oral)를 사용해 독파모 사업 차기 라운드 진출 모델 3종(LG K-EXAONE-236B, SKT A.X-K1, Upstage Solar-Open-100B)의 functional lineage를 reference 모델군과 비교한다.

## 디렉토리

```
.
├── README.md
├── pyproject.toml          # 로컬 분석 의존성
├── configs/
│   └── models.yaml         # 모델 → SageMaker 인스턴스 매핑
├── scripts/                # 추후: SageMaker 제출, 로컬 분석
├── cdk/                    # AWS 인프라 (TypeScript CDK)
│   ├── bin/llm-dna.ts
│   ├── lib/llm-dna-stack.ts
│   ├── package.json
│   ├── tsconfig.json
│   └── cdk.json
└── out/                    # 로컬 DNA 벡터 캐시 (gitignored)
```

## 환경 (us-east-1)

| 자원 | 상태 |
|------|------|
| AWS 계정 | 779411790546 (profile: roboco) |
| Region | us-east-1 |
| ml.p5.4xlarge spot quota | 4 (즉시 가용 — reference 소형) |
| ml.p5.48xlarge spot quota | 신청 PENDING (L-82733FAD, 1 instance) |
| ml.p5.48xlarge spot score | 9 (capacity 풍부) |

## 진행 단계

- [x] **Phase 0**: Mac 로컬 sanity check (distilgpt2 CPU 추론 확인)
- [x] **Phase 1**: AWS 사전 점검 (quota 신청 → 승인)
- [x] **Phase 2**: CDK 스택 배포 (S3, ECR, Secrets, IAM) — us-east-1
- [x] **Phase 3**: llm-dna 컨테이너 빌드 + ECR 푸시 — 10.68 GB, `llm-dna:latest`
- [ ] **Phase 4**: Reference 모델 DNA 추출 (ml.p5.4xlarge)
- [ ] **Phase 5**: Korean 3종 DNA 추출 (ml.p5.48xlarge)
- [ ] **Phase 6**: 로컬 phylogenetic 분석 + `docs/05-llm-dna-analysis.md` 작성

## 배포된 리소스 (us-east-1, 2026-04-26)

| Output | Value |
|--------|-------|
| ArtifactsBucketName | `llm-dna-779411790546-us-east-1` |
| ContainerRepoUri | `779411790546.dkr.ecr.us-east-1.amazonaws.com/llm-dna` |
| HfTokenSecretArn | `arn:aws:secretsmanager:us-east-1:779411790546:secret:llm-dna/hf-token-hdE2Hh` |
| SageMakerRoleArn | `arn:aws:iam::779411790546:role/LlmDnaSageMakerExecutionRole` |

## CDK 배포 (Phase 2)

```bash
cd cdk
npm install
npx cdk bootstrap aws://779411790546/us-east-1   # 1회만
npx cdk synth
npx cdk deploy

# HF token을 Secrets Manager에 주입 (~/.zshrc의 HUGGINGFACE_ACCESS_TOKEN 사용)
ARN=$(aws cloudformation describe-stacks --stack-name LlmDnaStack \
  --region us-east-1 \
  --query "Stacks[0].Outputs[?OutputKey=='HfTokenSecretArn'].OutputValue" \
  --output text)
aws secretsmanager put-secret-value \
  --secret-id "$ARN" \
  --secret-string "$HUGGINGFACE_ACCESS_TOKEN" \
  --region us-east-1
```

## 비용 추정

| 항목 | 추정 |
|------|------|
| Reference 7개 모델 (ml.p5.4xlarge spot, ~30분/모델) | ~$15 |
| GLM-5-FP8 + DeepSeek-V2.5 (ml.p5.48xlarge spot) | ~$60 |
| Korean 3종 (ml.p5.48xlarge spot, 1~2시간/모델) | ~$150 |
| S3 (cache + 출력) | <$5 |
| **합계** | **~$230** |
