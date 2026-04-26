---
title: "AWS CDK TypeScript 패턴"
type: concept
tags: [aws, cdk, typescript, iac, serverless]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: []
---

# AWS CDK TypeScript 패턴

본 프로젝트 인프라(`experiments/llm-dna/cdk/`)의 IaC 표준. **IaC=CDK(TS), spot+serverless 극대화** 원칙.

## 디렉토리 구조

```
cdk/
├── package.json          # aws-cdk-lib, constructs, ts-node
├── tsconfig.json
├── cdk.json              # app entry point (npx ts-node bin/llm-dna.ts)
├── bin/llm-dna.ts        # App + Stack 인스턴스화
└── lib/llm-dna-stack.ts  # Stack 정의
```

## 핵심 패턴

### 1. RemovalPolicy.RETAIN for stateful

```typescript
new s3.Bucket(this, 'Artifacts', {
  removalPolicy: cdk.RemovalPolicy.RETAIN,  // 실수로 stack destroy해도 데이터 보호
  encryption: s3.BucketEncryption.S3_MANAGED,
  blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
});
```

### 2. Lifecycle rule로 비용 통제

```typescript
lifecycleRules: [
  { id: 'expire-hf-cache', prefix: 'cache/', expiration: cdk.Duration.days(30) },
  { id: 'expire-incomplete-uploads', abortIncompleteMultipartUploadAfter: cdk.Duration.days(1) },
]
```

HF cache는 일시적이라 30일 만료. DNA 벡터(prefix `dna/`)는 만료 없이 영구 보관.

### 3. Secrets Manager로 token 관리

```typescript
const hfTokenSecret = new secretsmanager.Secret(this, 'HfToken', {
  secretName: 'llm-dna/hf-token',
});
hfTokenSecret.grantRead(sagemakerRole);
```

CDK는 secret value를 코드에 박지 않는다. 배포 후 `aws secretsmanager put-secret-value`로 별도 주입.

### 4. ECR 라이프사이클

```typescript
new ecr.Repository(this, 'Container', {
  repositoryName: 'llm-dna',
  imageScanOnPush: true,
  lifecycleRules: [{ description: 'Keep last 5', maxImageCount: 5 }],
});
```

### 5. CfnOutput으로 외부 통합

```typescript
new cdk.CfnOutput(this, 'ArtifactsBucketName', {
  value: artifactsBucket.bucketName,
});
```

이후 Python 스크립트에서:
```python
import boto3, json
cf = boto3.client('cloudformation')
outputs = cf.describe_stacks(StackName='LlmDnaStack')['Stacks'][0]['Outputs']
bucket = next(o['OutputValue'] for o in outputs if o['OutputKey'] == 'ArtifactsBucketName')
```

### 6. Tagging으로 비용 분리

```typescript
cdk.Tags.of(this).add('project', 'korea-ai-foundation-model-verification');
cdk.Tags.of(this).add('component', 'llm-dna');
```

Cost Explorer에서 tag 필터로 본 프로젝트 비용만 추출 가능.

## 워크플로우

```bash
cd experiments/llm-dna/cdk
npm install
npx cdk bootstrap aws://<acct>/us-east-1   # 1회
npx cdk synth                              # CloudFormation YAML 생성 (검증)
npx cdk deploy                             # 실제 배포
npx cdk destroy                            # RETAIN 리소스 제외 삭제
```

## Serverless 관점 강화

본 프로젝트는 명시적 Lambda는 없지만 다음이 모두 serverless 모델:
- **S3**: 사용량 기반
- **Secrets Manager**: 시크릿당 월 $0.40 + API 호출당 $0.05/10K
- **ECR**: 스토리지 GB당 + 데이터 전송
- **SageMaker training jobs**: 실행 초당, idle cost 0

**확장 옵션**: Step Functions로 다중 모델 배치 orchestration, EventBridge로 완료 알림 → 본 프로젝트 규모에선 과잉. Python 스크립트로 충분.

## 관련 페이지

- [[sagemaker-spot-training]] — 본 스택의 주 워크로드
- [[huggingface-hub-usage]] — Secret에 들어가는 token
