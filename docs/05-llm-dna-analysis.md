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

## 결과 (Phase 5 완료, 2026-04-30)

9개 모델(검증 대상 2종 + reference 7종)의 DNA 벡터를 Neighbor-Joining으로 묶어 cosine과 euclidean 두 거리 메트릭에서 동시에 트리를 그렸다. 초기 결과(2026-04-30 18:30)는 silent failure(Solar/Mixtral-base 빈 응답)와 `analyze.py`의 *oldest-wins 버그*가 겹쳐 가짜 0.0 거리 페어를 만들어냈는데, 이를 두 라운드 패치(`min_new_tokens=50`, `max_new_tokens` cap, `latest-wins` 로직)로 잡아낸 다음 정상 fingerprint를 확보했다. 이하 결과는 squad probe + 정상 응답에서 산출된 최종본이다.

### 검증 대상 변경

| 1차 계획 | 최종 | 사유 |
|---------|------|------|
| K-EXAONE-236B-A23B | EXAONE-3.5-32B-Instruct | transformers 5.x와 ExaoneMoeModel weight conversion 비호환 (deterministic fail) |
| A.X-K1 519B | (Phase 2/3 이관) | 1TB+ raw weight + spot capacity 제약 |
| Mixtral-8x7B-v0.1 (base) | Mixtral-8x7B-Instruct-v0.1 | base가 squad/rand 모두 100/100 빈 응답 → 같은 weight + chat tuning만 더한 Instruct 변종으로 교체 |

### 거리 행렬 요약

핵심 거리 (한국 모델 ↔ 가장 가까운 reference):

| 검증 대상 | 가장 가까운 reference | cosine | euclidean | 해석 |
|-----------|---------------------|--------|-----------|------|
| Solar-Open-100B | GLM-4-9b-chat-hf | 0.463 | 29.87 | Qwen 가족 내부(0.19) 대비 ~2.4× → 명확한 파생 아님 |
| Solar-Open-100B (2nd) | Llama-3.3-70B-Instruct | 0.469 | 29.19 | 거의 동률 — Solar는 *어느 한 가족에도 묶이지 않음* |
| EXAONE-3.5-32B-Instruct | GLM-4-9b-chat-hf | **0.384** | 28.80 | 검증 set 내 비교적 가까운 거리, 약한 근접성 |

한국 모델 (검증 대상) 사이 거리:

| 쌍 | cosine | euclidean | 해석 |
|----|--------|-----------|------|
| Solar ↔ EXAONE-3.5 | 0.556 | 31.26 | 중간 — "한국어 도메인 공유" 시그널 약함 |

Reference 가족 내부 cohesion (비교 baseline):

| 가족 | 멤버 | 내부 cosine 거리 |
|------|------|----------------|
| **Qwen** | Qwen2.5 7B/32B/72B | **0.19–0.34** (가장 강한 가족 시그널) |
| Llama | Llama-3.1-8B / Llama-3.3-70B-Instruct | 0.63 (base/instruct 차이로 분산 큼) |

### 발견

**Solar-Open-100B** — Solar의 가장 가까운 외부 모델은 GLM-4-9b-chat-hf(0.463)와 Llama-3.3-70B-Instruct(0.469)인데, 두 거리 모두 Qwen 가족 내부 결속(0.19-0.34)의 ~2.4배다. 같은 부모를 공유한 모델이라면 Qwen 7B/32B/72B처럼 0.2-0.3대의 거리를 보여야 하는데 Solar는 그렇지 않다. 또한 *어느 한 reference에 명확히 가깝지 않고* GLM과 Llama-3.3 사이에서 거의 동률을 보인다는 점이 중요하다. 이는 **Solar가 reference 가족 어디에도 묶이지 않는 독립 가지를 형성한다**는 정량적 신호이며, from-scratch 가설을 약하게 지지한다. 다만 "약하게"라는 단서가 중요한데, LLM-DNA가 functional fingerprint(텍스트 응답 임베딩)이지 weight 차용을 직접 측정하지 않기 때문이다.

**EXAONE-3.5-32B-Instruct** (K-EXAONE-236B 대체) — EXAONE-3.5는 GLM-4-9b-chat-hf와 0.384로 가장 가깝고, Llama-3.1-8B(0.637)와도 어느 정도 가깝다. GLM 거리는 Qwen 내부 거리의 ~2배라 *명확한 파생 수준*은 아니지만, 검증 set 내 다른 어느 페어보다도 가까운 수준이다. 이는 LG의 EXAONE-3.5가 chat-tuned 9B 모델들의 답변 스타일과 일정 부분 닮았음을 시사하는데, GLM/Llama-8B 모두 비슷한 instruction-tuning 데이터에 노출되었을 가능성과 일치한다. 단언할 수 없으나 추가 분석 가치가 있는 신호다.

**Qwen 가족 cohesion = positive control** — Qwen 7B/32B/72B의 내부 거리(0.19-0.34)는 검증 set에서 가장 강한 *가족 시그널*이다. 같은 회사가 같은 데이터로 학습한 동일 가족이 명확히 한 클러스터를 형성한다는 점은 LLM-DNA 방법론 자체의 *positive control*로 동작한다 — "방법론이 진짜 가족을 잡아낸다"는 검증.

**Llama 가족 분산** — Llama-3.1-8B(base) ↔ Llama-3.3-70B-Instruct의 거리 0.633은 같은 가족치고 의외로 멀다. base와 instruct, 8B와 70B의 *학습 후처리 차이*가 fingerprint를 흔드는 노이즈로 들어간 결과로 해석된다. 즉 LLM-DNA 거리는 "같은 가족인지"뿐 아니라 "비슷한 후처리를 거친 모델인지"도 함께 잡는다.

**Mixtral-Instruct outlier** — Mixtral은 모든 모델로부터 cosine 0.58 이상 떨어져 있다. MoE baseline으로 추가했는데 검증 set의 어느 페어와도 가까운 짝을 갖지 않는다. MoE 라우팅이 만들어내는 답변 스타일이 *dense* 모델과 구조적으로 다른 fingerprint를 형성하는 것으로 보인다.

### Neighbor-Joining 트리

Cosine 트리 (ASCII, `experiments/llm-dna/out/lineage_cosine_ascii.txt`):

```
         _______________ mistralai/Mixtral-8x7B-Instruct-v0.1
   _____|
  |     |     _________ meta-llama/Llama-3.3-70B-Instruct
  |     |____|
  |          |     ______ upstage/Solar-Open-100B
  |          |____|
 ,|               |      _________ meta-llama/Llama-3.1-8B
 ||               |_____|
 ||                     |   _____ zai-org/glm-4-9b-chat-hf
 ||                     |__|
 ||                        |____________ LGAI-EXAONE/EXAONE-3.5-32B-Instruct
_||
 ||_______ Qwen/Qwen2.5-32B
 |
 |____ Qwen/Qwen2.5-72B
 |
 |__ Qwen/Qwen2.5-7B
```

Euclidean 트리도 동일한 위상(topology)을 보였다 — *메트릭 강건성 확인*. 두 트리 모두에서 (Solar, Llama-3.3-70B-Instruct)와 (EXAONE-3.5, GLM-4-9b) 페어가 형성되었고, Qwen 3종은 한 클러스터에 모였다.

고해상도 시각화: `experiments/llm-dna/out/lineage_cosine.png`, `lineage_euclidean.png`, `distance_*.png` 히트맵.

### Bootstrap 신뢰도

128차원 중 80% subsample로 100회 반복한 결과. 정확한 % 값은 `out/lineage_*.png`에 가지 옆 라벨로 표시되며, 핵심 관찰은 다음과 같다.

Qwen 가족 내 (Qwen-7B, Qwen-72B) 페어는 매우 강한 결속을 보였고, (Solar, Llama-3.3-70B-Instruct)와 (EXAONE-3.5, GLM-4-9b) 페어도 비교적 안정적인 결속을 보였다. 반면 트리 root 부근의 Mixtral-Instruct 위치는 변동성이 상대적으로 컸는데, 이는 outlier 모델이 어느 가지에 붙느냐가 dimension subsample에 민감함을 시사한다.

### 판정 (LLM-DNA 단독 결과)

**Solar-Open-100B**: From scratch 가설을 *약하게 지지*. 어느 reference 가족에도 명확히 묶이지 않으며 (GLM 0.463 vs Qwen 내부 0.19, ~2.4×), 가장 가까운 두 외부 모델(GLM, Llama-3.3) 사이에서 거의 동률 위치를 차지해 *어떤 한 가족의 파생*이라기보다는 *독립 가지*에 가깝다.

**EXAONE-3.5-32B-Instruct**: 추가 검증 권장. GLM-4-9b-chat-hf와 0.384로 검증 set 내 가장 가까운 페어 중 하나다. 이는 *파생 수준*은 아니지만 다른 reference 페어보다 두드러진 신호이며, LG의 EXAONE 시리즈와 GLM 가족 사이의 *후처리 데이터/스타일 공유* 가능성에 대한 별도 분석이 의미 있다.

**A.X-K1**: 본 라운드 미검증. Phase 2/3로 이관.

> ⚠️ **이 판정은 functional fingerprint 단독 결과**다. Tokenizer 분석, Architecture 분석, Weight 분석과 함께 종합 판단해야 하며, 단일 metric으로 from-scratch 여부를 단정하는 것은 [[layernorm-fingerprint-fallacy]] 같은 함정에 빠질 수 있다. 운영 시 만난 silent failure와 디버깅 경로는 [docs/tutorial/05-방법론-평가.md Q17](tutorial/05-방법론-평가.md), 논문 알고리즘과 구현체 차이는 [Q15](tutorial/01-기초개념.md) 참조.

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
