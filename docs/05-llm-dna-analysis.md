# 05. LLM-DNA 계통 분석

> **상태**: 🚧 진행 중 (Phase 2 완료, Phase 3 컨테이너 준비 완료, Phase 4 sanity 잡 진행 중)
> **방법**: [LLM-DNA](https://github.com/Xtra-Computing/LLM-DNA) (ICLR'26 Oral) functional fingerprint 추출 + Neighbor-Joining 트리 구축
> **목적**: [[dokpamo-project]] 차기 라운드 진출 3종(Solar-Open-100B / K-EXAONE-236B-A23B / A.X-K1)의 functional 계통을 reference 모델군과 비교하여 from scratch 주장의 정황 증거 보강

---

## 분석 방법론

상세는 위키 참조: [[llm-dna-overview]] / [[dna-extraction-pipeline]] / [[neighbor-joining]] / [[distance-metrics]]

### 1. DNA 벡터 추출

각 모델에 대해 동일한 random 프롬프트 100개를 입력하고 출력 임베딩을 평균낸 뒤 random projection으로 128차원 DNA 벡터로 축소:

```bash
calc-dna --model-name <id> --dataset rand --max-samples 100 \
         --dna-dim 128 --reduction-method random_projection --random-seed 42
```

### 2. 거리 행렬

벡터 간 cosine distance + euclidean distance (multi-metric robustness check). 상세: [[distance-metrics]].

### 3. Neighbor-Joining 트리

unrooted 트리. 가법성·노이즈 강건성 보장. 상세: [[neighbor-joining]].

### 4. Bootstrap 신뢰도

128차원 중 80% 무작위 sub-sample을 N회 (기본 100) 반복하여 각 clade의 출현 빈도를 신뢰도(%) 로 표시. 70% 이상이면 robust.

---

## 분석 대상

### 검증 대상 (한국 모델 3종)

| 모델 | 파라미터 | 인스턴스 | 위키 |
|------|---------|---------|------|
| `upstage/Solar-Open-100B` | 102B MoE | ml.p5.48xlarge | [[solar-open-100b]] |
| `LGAI-EXAONE/K-EXAONE-236B-A23B` | 236B MoE (23B active) | ml.p5.48xlarge | [[k-exaone-236b]] |
| `skt/A.X-K1` | 519B MoE | ml.p5.48xlarge | [[ax-k1]] |

### Reference 모델군

| 그룹 | 모델 | 위키 |
|------|------|------|
| Llama | Llama-3.1-8B, Llama-3.3-70B-Instruct | [[llama-3-family]] |
| Qwen | Qwen2.5-7B / -32B / -72B | [[qwen-25-family]] |
| GLM | glm-4-9b-chat-hf, GLM-5-FP8 | [[glm-family]] |
| Mixtral | Mixtral-8x7B-v0.1 | [[mixtral-8x7b]] |
| DeepSeek | DeepSeek-V2.5 | [[deepseek-v25]] |

---

## 결과 (TBD — Phase 5 완료 후 채움)

### 거리 행렬 요약

> 자동 생성: `scripts/analyze.py` 실행 시 `experiments/llm-dna/out/distance_<metric>.{csv,png}` 생성

핵심 거리 (한국 모델 ↔ 가장 가까운 reference):

| 검증 대상 | 가장 가까운 reference | cosine | euclidean | 해석 |
|-----------|---------------------|--------|-----------|------|
| Solar-Open-100B | TBD | TBD | TBD | TBD |
| K-EXAONE-236B-A23B | TBD | TBD | TBD | TBD |
| A.X-K1 | TBD | TBD | TBD | TBD |

한국 모델끼리의 거리:

| 쌍 | cosine | euclidean | 해석 |
|----|--------|-----------|------|
| Solar ↔ K-EXAONE | TBD | TBD | TBD |
| Solar ↔ A.X-K1 | TBD | TBD | TBD |
| K-EXAONE ↔ A.X-K1 | TBD | TBD | TBD |

### Neighbor-Joining 트리

(이미지 삽입: `experiments/llm-dna/out/lineage_cosine.png`)

**관전 포인트**:
- [ ] 한국 모델 3종이 어느 reference 그룹에 가장 가까운가
- [ ] 한국 모델끼리 자체 clade를 형성하는가, 아니면 각자 다른 reference 군에 흩어지는가
- [ ] [[solar-open-100b]]가 [[glm-family]]에 가까운가 (논란 검증)
- [ ] [[k-exaone-236b]]가 [[deepseek-v25]] (236B)와 가까운가 (스케일 영향)
- [ ] [[ax-k1]]이 어느 reference와도 멀리 떨어진 위치에 있는가

### Bootstrap 신뢰도

| Clade | Cosine support | Euclidean support |
|-------|---------------|-------------------|
| TBD | TBD% | TBD% |

---

## 판정 가이드 (사후 채울 것)

### From scratch 강한 정황 증거가 되는 결과

1. 한국 모델 3종이 reference 모든 그룹에서 충분히 멀리 떨어진 위치에 있음
2. 한국 모델끼리는 reference보다 가까움 (한국 PRD/도메인 데이터 공유 시그널)
3. Bootstrap 신뢰도 70% 이상

### 추가 검증이 필요한 결과

1. 어느 한국 모델이 특정 reference clade 내부에 포함됨 (Solar→GLM, K-EXAONE→DeepSeek 등)
2. Cosine과 Euclidean 트리 위상이 크게 다름 (메트릭 의존성)
3. Bootstrap 신뢰도 50% 미만

### 본 프로젝트의 한계

- LLM-DNA는 functional similarity만 측정 — weight 직접 차용 단정 불가
- 동일 BPE 코퍼스로 from scratch 학습한 두 모델도 비슷한 DNA 가능
- 305 모델 중 일부만 비교한 것이므로 베이스라인이 좁음
- 상세는 [[layernorm-fingerprint-fallacy]], [[model-provenance-testing]] 참조

---

## 인프라 메타데이터

| 항목 | 값 |
|------|-----|
| Region | us-east-1 |
| Instance | ml.p5.48xlarge spot (sanity test) / configured ml.p5.4xlarge for reference + p5.48xlarge for target |
| Container | `779411790546.dkr.ecr.us-east-1.amazonaws.com/llm-dna:latest` (10.68 GB) |
| Stack | `LlmDnaStack` (CDK TypeScript) |
| 코드 | `experiments/llm-dna/` |

상세는 위키 참조: [[sagemaker-spot-training]] / [[aws-cdk-typescript]] / [[huggingface-hub-usage]]

---

## 진행 이력

| 날짜 | 단계 | 비고 |
|------|------|------|
| 2026-04-26 | Phase 0 | 로컬 sanity (distilgpt2 CPU) 검증 완료 |
| 2026-04-26 | Phase 1 | us-east-1 spot quota 신청 → 즉시 승인 (p5.48xlarge=1) |
| 2026-04-26 | Phase 2 | CDK 스택 배포 (S3, ECR, Secret, IAM) |
| 2026-04-26 | Phase 3 | 컨테이너 빌드 + ECR push (10.68 GB, ~70분) |
| 2026-04-26 | Phase 4 sanity | Qwen2.5-7B 잡 제출 → p5.4xlarge spot capacity 부족 (score 1) |
| TBD | Phase 4 | Reference 9개 — capacity 회복 또는 p5.48xlarge fallback 후 진행 |
| TBD | Phase 5 | Korean 3종 (p5.48xlarge) |
| TBD | Phase 6 | analyze.py로 트리 구축 + 본 보고서 결과 섹션 채움 |
