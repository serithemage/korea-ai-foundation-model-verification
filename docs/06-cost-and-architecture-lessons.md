# 06. 비용 분석과 아키텍처 교훈 (LLM-DNA Phase 4–6 회고)

> **상태**: ✅ 정리 완료 (2026-04-30)
> **목적**: LLM-DNA 1라운드 검증을 마친 후 SageMaker 잡 77건의 비용 데이터를 정량 분석하고, 다음 라운드(A.X-K1, K-EXAONE-236B 재시도)에 그대로 적용 가능한 운영 교훈을 정리한다. 후속 검증자가 같은 학습 곡선을 다시 밟지 않도록 *재사용 가능한 형태*로 기록하는 것이 목적이다.

---

## 비용 총액

2026-04-26 ~ 2026-04-30 기간 동안 LLM-DNA 추출을 위해 SageMaker에서 실행한 잡은 총 77건이고, billable 시간은 13.5h, 비용은 spot pricing 기준 **약 $64.28**이었다. 이 숫자만 보면 작아 보이지만 그 안에 두 가지 다른 이야기가 섞여 있다 — 검증 자체에 들어간 *유효 비용*과 시행착오에서 흘러나간 *학습 곡선 비용*이다.

| 분류 | 잡 수 | Billable | 비용 (spot) | 비중 |
|---|---|---|---|---|
| **Completed** (성공) | 22 | 7.61h | **$30.35** | 47.2% |
| Failed | 42 | 3.41h | $20.23 | 31.5% |
| Stopped (수동 중단) | 13 | 2.52h | $13.71 | 21.3% |
| **합계** | 77 | 13.5h | **$64.28** | 100% |

성공한 22건은 reference 모델 7종 + EXAONE-3.5-32B-Instruct + Mixtral-Instruct + Solar-Open-100B의 추출 + 일부 재시도가 포함된 결과다. 이 22건이 만든 비용 $30.35가 본 검증의 *실질 비용*이고, 나머지 $33.93(52.8%)이 학습 곡선에 들어간 비용이다.

인스턴스별로 쪼개면 어디에 돈이 모였는지가 더 분명해진다.

| 인스턴스 | 잡 수 | Billable | 비용 | 시간당 단가 (spot 추정) |
|---|---|---|---|---|
| ml.p5.48xlarge (8×H100) | 8 | 1.44h | **$28.86** | ~$20/h |
| ml.g7e.12xlarge (4×L40S) | 39 | 7.66h | $26.81 | ~$3.5/h |
| ml.g7e.4xlarge (1×L40S) | 21 | 4.02h | $4.82 | ~$1.2/h |
| ml.g7e.48xlarge (8×L40S) | 6 | 0.42h | $3.79 | ~$9/h |
| ml.p5.4xlarge | 3 | 0.00h | $0.00 | (capacity 즉시 거부) |

가장 작은 잡 수(8건)에 가장 큰 비용($28.86)이 몰린 p5.48xlarge가 흥미롭다. 이는 *비싼 인스턴스에서 짧게 실패한 잡들*이 비용을 끌어올리는 패턴이다 — 나중에 보겠지만 K-EXAONE-236B의 deterministic 실패 4회와 Solar 1차 잡의 stopped 1건이 여기 모여 있다.

비교 기준점으로, on-demand 가격으로 환산하면 같은 13.5h가 약 $200+가 된다. spot 사용으로 절감된 금액이 약 $135 (~68%)이며, 이 절감은 본 프로젝트의 *가장 큰 단일 인프라 결정*이었다.

---

## 낭비 해부 — 어디서 새었나, *왜* 사전에 막을 수 없었나

낭비 비용 $33.93의 약 70%가 네 카테고리에 집중된다. 각 카테고리는 *왜 이런 학습 곡선이 필연적이었는지*와 *다음에는 어떻게 막을 것인지*가 짝을 이룬다.

**1. K-EXAONE-236B-A23B 본체 시도 — 약 $13** (낭비의 38%). p5.48xl에서 4회, g7e에서 4회, 총 8회 반복 시도. transformers 5.x의 ExaoneMoeModel weight conversion이 *deterministic하게* 실패했는데, 매 시도가 718-shard 다운로드(7-9분 billable)까지 진행한 다음 conversion 단계에서 같은 에러로 폭발했다. 첫 번째 실패 로그를 정밀하게 분석해 "이건 deterministic이라 재시도가 의미 없다"는 결론을 내리기까지 약 $10이 들었다. 이건 단일 가장 큰 학습 비용이며, 다음 라운드의 첫 번째 교훈이다 — *실패가 reproducible하면 두 번째 시도 후 즉시 격리한다*. custom 아키텍처(trust_remote_code=True)는 별도 sanity 잡으로 분리해 *큰 잡에 들어가기 전*에 모델 로드 단계만 검증하는 패턴이 필요하다.

**2. Solar 1차 잡 stopped — $7.93** (낭비의 23%). p5.48xl에서 22/100 prompt 진행 후 200s/prompt라는 비정상 속도를 발견하고 수동 stop. *Mixtral의 35s/prompt와 비교하는 중간 폴링이 없었다면* 그대로 4.5시간 더 진행해 $135까지 갈 수 있었다. 다행히 25분 만에 잡았지만, 이는 사람이 우연히 비교해서 발견한 것이지 시스템이 자동으로 잡은 것은 아니다. 다음 라운드 교훈 — *실시간 generation 속도 알림*이 필요하다. CloudWatch metric이나 entrypoint에서 첫 5개 prompt의 평균 속도가 expected ceiling을 넘으면 자동으로 잡을 죽이는 메커니즘이 있으면 같은 비용을 $1 안에 차단할 수 있다.

**3. Initial setup 단계 g7e.12xl 잡들 — 약 $6** (낭비의 18%). 04-27 단일 날에 yaml/CDK 디버깅 단계에서 발생한 짧은 실패 잡들이다(각 5-8분 billable). 컨테이너 boot + HF 인증 + multi-GPU dispatch까지 진행한 다음 yaml 오타나 secret 누락으로 fail. 이건 *환경 변수 + 인증의 통합 검증*이 부재했던 결과다. 로컬 sanity(distilgpt2 CPU, $0)에서 코드 자체는 검증했지만, *실제 SageMaker 환경의 secret + IAM 경로*는 검증하지 못했다. 다음 라운드 교훈 — *최소 권한 sanity 잡* 한 번을 가장 작은 인스턴스(g7e.4xl, $1/h)에서 distilgpt2로 통과시킨 후에야 본 잡을 제출하는 게이트를 두는 것.

**4. Solar g7e.48xl 4-bit 시도들 — 약 $2** (낭비의 6%). llm-dna 0.2.x의 quantization 자동 활성화(7B 이상 모델에 8-bit auto-enable)와 ModelWrapper의 single-GPU 강제가 충돌하는 문제. fp16 + multi-GPU 패치를 알기 *전*에 만난 함정이라 사전 차단은 어려웠지만, 일단 한 번 만나고 나면 모든 100B+ 모델에 동일하게 적용되는 패턴이라 패치를 entrypoint에 굳혀 두면 다시는 발생하지 않는다.

남은 30%는 위 네 카테고리에 안 잡히는 작은 시행착오들이다. 가장 큰 항목은 Llama-3.3-70B-Instruct stopped 1건($3.88) — g7e.12xl에서 66분 진행 후 capacity issue로 사용자가 중단한 사례이고, 같은 모델이 다음 시도에서 정상 완료됐다.

---

## 향후 검증에 사용할 *재사용 가능한* 아키텍처 교훈

비용/낭비 분석에서 도출한 교훈을 *코드와 컨벤션으로 굳힐 수 있는 형태*로 정리한다. 추상적인 "다음에는 신중하게"가 아니라, 다음 라운드 entrypoint와 yaml에 곧바로 들어갈 수 있는 변경들이다.

### Sanity job 단계를 entrypoint에 통합

현재 `dna_train.py`는 다운로드 → 모델 로드 → calc-dna 100 prompt 추출 → 결과 업로드 순서다. 여기에 *모델 로드 직후, 본 추출 전*에 Phase 0 게이트를 끼워 넣으면 거의 모든 함정을 단가 $0.50의 1분 사전 검증으로 잡을 수 있다.

```python
# 본 추출 직전에 1개 prompt로 forward 1회 검증
test_response = model.generate("Hello, can you respond?", max_new_tokens=50)
if len(test_response.strip()) < 20:
    sys.exit(2)  # SageMaker가 Failed로 마킹 → spot 재시도 자동
```

이 한 가지 변경으로 (1) g7e의 grouped_mm RuntimeError, (2) base 모델 빈 응답, (3) 모델 로드 실패가 모두 1분 안에 잡힌다. K-EXAONE 같은 weight conversion 실패도 모델 로드 단계에서 즉시 잡히므로 718-shard 다운로드 비용이 없다.

### `max_new_tokens` cap을 default로 굳히기

이번 검증의 가장 강력한 단일 최적화. fingerprint 손실 없이 generation 시간 8× 단축. sentence-encoder의 ~512 토큰 컷오프가 결정적인 이유인데, 이를 모르면 max_new_tokens=2048(llm-dna default)이 *낭비*임을 알 수 없다. 다음 라운드부터는 `configs/models.yaml`의 `dna_extraction` 섹션에 `max_new_tokens: 256`을 default로 명시해 모든 모델에 적용해야 한다.

### 결과 매칭 컨벤션 — *latest-wins이 처음부터 default*

S3는 append-only이고 잡 prefix는 timestamp가 들어가므로, *항상 max(prefix)*로 매칭하는 것이 model_id별 latest를 자동 선택한다. 이번 검증의 `analyze.py`가 oldest-wins로 시작했고 옛 빈 응답 결과가 정상 결과를 가리는 silent bug를 만들었다. 다음 분석 스크립트들도 이 컨벤션을 따라야 하며, 하나의 helper 함수로 추출해 재사용해야 한다.

### 멀티 리전 stack 배포는 capacity가 아니라 *quota* 분산이 본질

spot placement score가 9여도 SageMaker quota=0이면 못 쓴다. us-east-1 spot quota=1, us-west-2 spot quota=0인 사례가 그 증거. 처음부터 두 리전에 stack을 배포하고, 각 리전의 quota를 미리 확인한 다음 `submit_dna.py --region`으로 라우팅하는 패턴은 검증된 아키텍처다. CDK에서 `roleName` 같은 글로벌 unique 자원은 제거해 두 리전 동시 배포가 가능하도록 했다.

### GPU 호환성 매트릭스를 config로 명시

transformers 5.7.0 + MoE forward = `torch._grouped_mm` (Hopper 9.0 전용)이라는 사실을 *알고 있는 상태*면 잡 제출 전에 차단할 수 있다. `configs/models.yaml`에 모델별 *최소 compute capability* 필드를 추가하고 `submit_dna.py`가 인스턴스의 GPU CC를 검증하면 5분 만에 막을 수 있다.

```yaml
- id: mistralai/Mixtral-8x7B-Instruct-v0.1
  min_cuda_capability: 9.0   # MoE grouped_mm requires Hopper
  instance_type: ml.p5.48xlarge
```

### Cold start fixed cost를 의식한 batch 추출

P5는 cold start 9분 + weights load 7-15분이 무료 시간이다. 잡당 generation이 짧으면 *고정비*가 변동비보다 크다. 이번 검증에서는 모델당 1잡이라 9개 모델에 9× cold start 비용이 들었다. 작은 모델 여러 개를 같은 잡에서 순차 추출(같은 컨테이너 + 같은 인스턴스)하면 cold start cost를 amortize할 수 있다 — 다음 라운드에서 7B 이하 reference 모델들을 *batched 잡 1개*로 묶으면 setup 비용을 67% 줄일 수 있다.

### Quantization과 device_map의 상호작용은 명시적으로 결정

llm-dna 0.2.x의 `--device auto` + 자동 8-bit는 7B 이상 모델에서 충돌을 일으킨다(single-GPU 강제 + bnb CPU offload 거부). 본 프로젝트는 `--no-quantization` 플래그 + multi-GPU 패치로 우회했지만, *모든 모델 크기에서 일관된 동작*을 보장하려면 entrypoint에서 모델 크기에 따라 자동 fp16/4bit 선택 + device_map 결정을 *명시적*으로 해야 한다. yaml의 `load_in` 필드를 그대로 따르되, llm-dna 내부 자동 quantization을 항상 비활성화하는 것이 시작점이다.

### `--continue-on-error`는 검증 통과 조건과 함께 사용

이 옵션 자체는 유용하다(일부 prompt 실패 시 부분 결과라도 건짐). 그러나 *모든* prompt가 실패할 때도 잡을 Completed로 보고하는 양면성이 silent failure의 한 축이었다. Entrypoint에서 *통과 임계값*(예: nonempty ratio > 0.9)을 검증해 미만이면 fail로 마킹해야 한다. 그러면 SageMaker가 잡을 Failed로 표시하고 spot 재시도까지 받을 수 있어 일거양득이다.

### 직관에 반하는 결과를 자동 의심하는 단계가 분석 파이프라인에 필요

"Solar와 Mixtral-base 거리 0.0"이라는 결과는 *알고리즘적으로 가능*하지만 *물리적으로 의심스러운* 결과였다. 이번에는 사람이 의심하고 sanity check를 돌렸지만, 분석 파이프라인 안에 *모든 페어 거리의 분포에서 outlier 감지*(예: 0.1 미만 페어 자동 플래그)를 두면 같은 함정을 자동으로 잡는다. `analyze.py`에 1줄 검증이면 충분하다.

### Spot quota는 placement score보다 strict — 사전 검사 필수

이번 검증에서 us-east-1 quota=1 때문에 Solar 잡과 Mixtral 잡을 *직렬*로 실행해야 했고, 이 직렬화가 1.5h의 추가 wall time을 만들었다. `submit_dna.py`가 잡 제출 전에 `service-quotas`로 quota를 확인하고, 부족하면 다른 리전 fallback을 자동으로 시도하는 게 다음 라운드에서 추가할 만한 변경이다.

---

## 다음 라운드 적용 시 예상 절감

위 9개 교훈을 적용한다고 가정하면, 다음 라운드(A.X-K1 519B + K-EXAONE-236B 재시도)의 비용 구조는 어떻게 바뀔까. 가장 큰 변화는 두 곳에서 온다.

첫째, K-EXAONE 같은 *weight conversion 실패*는 sanity job 게이트(< $1)에서 잡혀, 718-shard 다운로드 후 실패하던 7-9분 × 8회 = $13의 학습 비용이 $4 이내로 줄어든다 (transformers 호환성 회복 후 첫 통과 + 한 번의 재시도). **약 $9 절감.**

둘째, *cold start amortization*과 *max_new_tokens cap*이 결합하면 잡당 wall time이 절반 이하로 줄어든다. p5.48xl 잡당 약 30분 → 15분이 되면 spot 비용도 절반 이하 ($10 → $5). 9개 모델 × 잡당 절감 $5 = **약 $45 절감** (단, 이번 라운드 reference 7종은 이미 끝났으므로 새 모델만 적용).

A.X-K1 519B는 *새로운 도전*이다. 1TB+ raw weight + multi-node 가능성. 이 영역의 비용은 여전히 예측 어렵지만, 지금까지의 교훈을 적용하면 *학습 곡선 비용은 0에 가깝게* 만들 수 있고 *유효 비용만 들이는* 잡을 한 번에 통과시킬 수 있을 것이다.

---

## 진행 이력

| 날짜 | 단계 | 비고 |
|------|------|------|
| 2026-04-30 | Phase 6 분석 완료 | 9개 모델 NJ 트리 + bootstrap 완료 ([docs/05](05-llm-dna-analysis.md)) |
| 2026-04-30 | 회고 정리 | SageMaker 잡 77건 비용 데이터 집계, 9개 아키텍처 교훈 도출 (본 문서) |
