# AWM·GhostSpec 기반 로컬 시범 검사

검사일: 2026-10-12. 진출 3개 모델의 공개 가중치에 추가 검사를 수행했다. **원 논문의 전체 실험을 재현한 결과가 아닌, 제한된 행·층의 시범 검사**다. AWS 자원이나 GPU를 사용하지 않았다. 이번 AWS 비용은 **$0**이다.

## 실행한 범위

| 모델 쌍 | AWM 기반 검사 | GhostSpec 기반 검사 |
|---|---|---|
| Solar Open 2 ↔ Solar Open 1 | embedding 표본으로 좌표·부호 정렬, 선택 Q/K의 UCKA | 양쪽 layer 0의 GQA QK 곱 스펙트럼 |
| K-EXAONE 2.0 ↔ K-EXAONE | embedding 표본으로 좌표·부호 정렬, 선택 Q/K의 UCKA | 신규 layer 5와 선행 layer 3의 global GQA QK 곱 스펙트럼 |
| A.X K2 ↔ A.X K1 | embedding 표본으로 좌표·부호 정렬 | MLA/SGA·FP8 attention adapter 미구현으로 미수행 |

원 개발사 리비전은 [기존 예비검증](07-second-round-analysis.md)과 동일하게 고정했다. JSON에서 직접 byte range를 읽었으며 원격 모델 코드를 실행하거나 모델 전체를 메모리에 올리지 않았다. 선택 가중치와 헤더를 413,794,320 bytes(약 414MB) 읽었다. 실행 시간은 세 쌍 합계 약 141초였으며, 이는 이번 CPU 시범 실행의 시간이다. 전체 모델 검사나 GPU 실행 시간으로 외삽하지 않는다.

## embedding 정렬: 검사에 쓰지 않은 행으로 확인

각 모델의 embedding에서 ID `4096~4223`, `32768~32895`, `65536~65663`, `98304~98431`의 총 512행을 읽었다. 모든 선택 ID의 토큰 문자열이 양쪽에서 같음을 확인했다.

선택 행의 짝수 위치 256개로 embedding 열의 cosine을 비교하고 Linear Assignment Problem을 풀었다. permutation과 부호를 복원한 뒤 **정렬에 사용하지 않은 홀수 위치 256개**에서 열별 cosine의 평균을 계산했다. 이 수치는 AWM의 최종 attention UCKA와 구분한다. 표본은 고정된 위치이며 무작위 대표 표본이라고 주장하지 않는다.

| 모델 쌍 | 정렬 학습 행 평균 cosine | 미사용 행: 정렬 전 | 미사용 행: 정렬 후 | 동일 좌표로 대응한 비율 |
|---|---:|---:|---:|---:|
| Solar 2 ↔ Solar 1 | 0.5696 | 0.5858 | 0.5859 | 99.9512% |
| EXAONE 2 ↔ EXAONE | 0.6591 | 0.6550 | 0.6550 | 100% |
| A.X K2 ↔ K1 | 0.2531 | 0.0010 | 0.0059 | 0.0140% |

Solar·EXAONE의 표본은 거의 같은 좌표에 대응했다. A.X는 정렬에 사용한 행에서는 0.2531까지 맞춰졌지만 새 행에서는 0.0059였다. **정렬 과정이 맞춰 낸 유사성을 곧바로 계승 증거로 해석하면 안 된다**는 차이를 보여준다.

단, 토큰 데이터의 독립성을 완전히 보장한 학습/평가 분리는 아니다. 인접 토큰 및 vocab 분포의 영향이 남는다. A.X의 낮은 수치는 “선택한 K1 embedding 표본의 정렬 후에도 일관된 상관을 찾지 못했다”는 뜻이며, 외부 모델 차용 부재나 전체 무작위 초기화의 증명은 아니다.

## 선택 attention 가중치의 UCKA

AWM 공식 구현의 U-centered linear CKA 수식을 사용했다. parent의 hidden 축을 위 permutation·부호로 맞추고 Q/K 행렬을 전치해 비교했다. 선택 Q/K **행렬 전체**를 읽었지만, 모든 layer를 검사한 것은 아니다.

| 모델 쌍 / 선택 layer | Q UCKA | K UCKA |
|---|---:|---:|
| Solar 신규 0 ↔ 선행 0 | 0.4861 | 0.3791 |
| EXAONE 신규 5 ↔ 선행 3 | 0.8328 | 0.8306 |
| A.X K2 ↔ K1 | 미수행 | 미수행 |

EXAONE의 두 선택 layer는 모두 global attention이다. 공개된 확장 설명을 참고한 탐색 비교이며 실제 초기화 layer 매핑을 복원했다는 뜻은 아니다. 두 쌍은 개발사가 계승을 공개한 사례다. **관련 없는 실제 모델 대조군의 점수 분포를 아직 만들지 않아 유의수준·판정 임계값을 설정하지 않았다.**

## GQA QK 곱의 스펙트럼

GhostSpec 공식 구현에서 사용하는 GQA 반복 후 `Q.T @ K`의 특이값을 계산했다. 실제 모델은 `head_dim=128`인데 `hidden_size / query_heads`는 Solar 64, EXAONE 96이다. 따라서 공식 구현이 사용하는 차원 추정값을 그대로 넣으면 K reshape가 맞지 않는다. 검사에서는 config의 명시적 `head_dim`과 실제 tensor shape를 확인했다.

거대한 dense 곱을 만들 필요는 없다. 같은 K head를 공유하는 Q head들을 합쳐 A를 구성하면 곱은 `A.T @ K`가 된다. `A.T=U R_A`, `K.T=V R_K`의 thin QR를 구하면 곱의 **0이 아닌 특이값**은 `R_A @ R_K.T`의 특이값과 같다. 이 항등식은 작은 GQA 행렬에서 실제 dense 곱과 대조했다.

| 선택 layer 쌍 | 스펙트럼 cosine | 스펙트럼 상대 L2 차이 |
|---|---:|---:|
| Solar 신규 0 ↔ 선행 0 | 0.9696 | 2.9815 |
| EXAONE 신규 5 ↔ 선행 3 | 0.9724 | 0.5096 |

두 지표는 특이값 벡터의 단순 기술 통계다. GhostSpec의 최종 POSA/MSE/Correlation 판정 점수가 아니다. cosine이 높아도 크기는 크게 달라질 수 있다. 이 값만으로 계승 여부를 판정하지 않는다. 이번에는 전체 layer 정렬(POSA), VO 곱, effective-rank 기반 최종 scoring을 실행하지 않았다.

## 방법별 상태와 다음 작업

- **AWM**: 표본 embedding 정렬과 단일 층 UCKA 시범을 수행. 전체 vocab·layer 매핑과 독립 모델 대조는 미수행.
- **GhostSpec**: 선택 GQA 층의 QK 스펙트럼 시범을 수행. 전체 fingerprint/POSA 재현은 미수행.
- **REEF**: 모델 forward 및 activation 수집이 필요. 이번에는 실행하지 않음.
- **Model Provenance Testing·LLMPrint**: 실제 대상·부모·독립 대조군의 query 환경과 추론 비용 산정이 먼저. 이번에는 실행하지 않음.
- **HuRef·PoL·PoTD**: 개발사 측 지문 또는 초기화·학습 transcript 접근이 필요한 범위는 공개 자료만으로 수행하지 못함. 개발사에 메시지를 보내지 않음.

우선 결론은 기존 공개 자료 판단을 유지한다. Solar·EXAONE의 관측값은 알려진 자체 계승 경로와 함께 검토할 추가 증거다. A.X의 이번 검사도 독립 확증을 제공하지는 않는다.

## 재현 자료

- [실행 결과 JSON](../wiki/raw/2026-10-12_weight-provenance-pilot.json): 리비전, token IDs, permutation·부호, 특이값, 읽은 범위·해시, 실행 시간.
- [실행 스크립트](../experiments/llm-dna/scripts/weight_provenance_probe.py).
- [수학 검증 3개](../experiments/llm-dna/scripts/test_weight_provenance_probe.py): 미사용 행의 permutation·부호 복원, UCKA의 feature 회전 불변성, 표준 비율과 다른 head 차원을 갖는 GQA의 compact/dense 특이값 일치.
- 검토한 공식 코드: [AWM 고정 commit](https://github.com/LUMIA-Group/AWM/tree/bc20ff8e63cec57f5da422ae065686ced275e76d), [GhostSpec 고정 commit](https://github.com/DX0369/GhostSpec/tree/1a2a8275a46a198a23b0091bde685c71aaa0e84b).

```bash
uv venv /tmp/provenance-probe-venv
uv pip install --python /tmp/provenance-probe-venv/bin/python numpy==2.5.4 scipy==1.18.1
/tmp/provenance-probe-venv/bin/python experiments/llm-dna/scripts/test_weight_provenance_probe.py
/tmp/provenance-probe-venv/bin/python experiments/llm-dna/scripts/weight_provenance_probe.py \
  --evidence wiki/raw/2026-10-12_second-round-evidence.json \
  --cache /tmp/provenance-probe-cache \
  --output /tmp/provenance-probe-results.json
```

캐시와 대용량 가중치 파일은 커밋하지 않는다. GPU 확대 단계의 범위·예산은 [별도 Spot 견적](10-spot-verification-estimate.md)에 있다. 승인 전 AWS 실행은 하지 않는다.
