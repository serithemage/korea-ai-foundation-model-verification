# EC2 Spot GPU 가중치 확대 검사

검사일: 2026-10-12. 사용자 승인 한도는 **$15, 세금 제외**다. 오하이오 `use2-az2`의 `g6e.12xlarge`를 **EC2 Spot 전용**으로 요청했고 `InstanceLifecycle=spot`을 확인했다. On-Demand 전환은 없다. 프로그램은 종료 코드 0과 `complete=true`로 끝났고 계산 구간은 **318.65초**였다. 부팅·환경 초기화 시간은 이 계산 시간에 포함하지 않는다.

## 무엇이 확대됐나

기존 CPU 시범은 같은 세대 쌍의 embedding 512행과 단일 GQA 층을 살폈다. 이번에는 다음을 추가했다.

- 대상과 선행 모델 6개에 Qwen2.5-7B Base·Instruct 2개를 더해 **8개 공개 모델**을 검사했다.
- 모델 쌍마다 공통 vocab에서 special token을 제외하고 **2,048행**을 골랐다. 1,024행으로 좌표·부호를 정렬하고 정렬에 쓰지 않은 1,024행으로 확인했다.
- 호환되는 모델 6개에서 **4개 GQA 층씩**, 총 **24개 QK 스펙트럼**을 추출했다.
- GQA 비교가 가능한 5쌍에서 4×4의 **80개 layer 쌍**을 모두 검사해 Q/K UCKA 160개를 기록했다.
- A.X K2·K1은 embedding만 검사했다. MLA/SGA·FP8 attention을 GQA와 같은 방식으로 검사한 결과는 없다.

전체 모델 추론이나 재학습은 수행하지 않았다. 모델 전체를 추론 메모리에 올리지 않고 필요한 embedding·Q/K만 읽었다. 가중치·헤더 읽기 합계는 **15,620,767,600 bytes(약 15.62GB)**다. CPU RAM이 큰 Spot 인스턴스에 원본을 보관하고 GPU 0에서 행렬 계산을 수행했다. 다른 GPU에서 모델을 병렬 추론한 작업은 아니다.

## 비교군의 의미

[Qwen2.5-7B-Instruct 공식 카드](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)는 instruction-tuned 모델임을 설명한다. Base→Instruct 쌍은 알려진 후속 학습의 양성 대조로 사용했다. Solar·EXAONE의 이전 세대 쌍은 개발사가 공개한 자체 계승 사례다.

대상 모델과 Qwen Base의 비교는 **무관 모델이라고 가정한 비교 후보**다. Qwen을 비롯한 모든 모델의 무작위 초기화 이력을 독립 검증했다는 뜻은 아니다. 음성 대조 후보가 한 family뿐이고 후보 수가 적으므로, 오탐률·p-value·판정 임계값을 추정하지 않는다.

각 모델 쌍마다 공통 어휘 집합이 다르다. 같은 후보와 비교해도 token 표본과 hidden 차원 정렬이 달라지므로 점수를 완전히 동일한 조건의 확률이나 전역 거리로 취급하지 않는다.

## embedding: 정렬에 쓰지 않은 표본의 결과

공통 토큰을 문자열 순서로 정렬하고 균등한 인덱스 간격으로 2,048개를 선택했다. 선택 배열의 짝수 위치 1,024개로 LAP를 풀고, 홀수 위치 1,024개에서 열별 cosine의 평균을 계산했다. 모든 표본의 token 문자열을 양쪽 vocab ID로 매핑했다. hidden 크기가 다르면 작은 쪽 차원 수만큼 직사각형 LAP로 대응했다.

| 대상 ↔ 비교 모델 | 정렬에 사용한 행 | 정렬에 쓰지 않은 행 | 해석 |
|---|---:|---:|---|
| Solar 2 ↔ Solar 1 | 0.5479 | **0.5481** | 선택 embedding에서 계승과 부합하는 상관이 유지됨 |
| EXAONE 2 ↔ EXAONE 1 | 0.6582 | **0.6539** | 선택 embedding에서 계승과 부합하는 상관이 유지됨 |
| A.X K2 ↔ K1 | 0.1333 | **0.0119** | 표본에서 일관된 K1 embedding 계승 신호를 찾지 못함 |
| Qwen Instruct ↔ Qwen Base | 0.9999 | **0.9999** | 알려진 후속 학습 쌍에서 높은 상관 확인 |
| Solar 2 ↔ Qwen Base | 0.1353 | **0.0352** | 선택한 외부 비교 후보와 낮은 상관 |
| EXAONE 2 ↔ Qwen Base | 0.1449 | **0.0415** | 선택한 외부 비교 후보와 낮은 상관 |
| A.X K2 ↔ Qwen Base | 0.1321 | **0.0170** | 선택한 외부 비교 후보와 낮은 상관 |

이 값은 AWM의 최종 attention UCKA가 아니다. embedding 정렬의 미사용 행 검사다. 서로 인접한 vocab 항목의 관계가 남을 수 있어 엄격히 독립적인 데이터 분리라고 주장하지 않는다. 기존 CPU 시범의 고정 ID 표본과 선택 방식도 달라 점수 차이를 모델의 변화로 해석하지 않는다.

## 선택 attention 층의 AWM 기반 UCKA

embedding의 LAP 대응과 부호를 적용한 뒤 Q/K를 전치해 U-centered linear CKA를 계산했다. 후보의 층이 일치한다고 가정하지 않고 선택한 4개 층끼리 전부 비교했다. 아래는 **사전에 선택한 같은 순서의 층 쌍**이다. 전체 4×4 행렬은 증거 JSON에 있으며, 아래 숫자는 전체 모델의 평균이 아니다.

| 모델 쌍 | 신규 층 / 선행 층 | Q UCKA | K UCKA |
|---|---|---:|---:|
| Solar 2 / Solar 1 | 0 / 0 | 0.4861 | 0.3791 |
| Solar 2 / Solar 1 | 4 / 4 | 0.2324 | 0.3324 |
| Solar 2 / Solar 1 | 20 / 20 | 0.4984 | 0.5580 |
| Solar 2 / Solar 1 | 44 / 44 | 0.4477 | 0.5236 |
| EXAONE 2 / EXAONE 1 | 5 / 3 | 0.8328 | 0.8306 |
| EXAONE 2 / EXAONE 1 | 25 / 15 | 0.6280 | 0.6143 |
| EXAONE 2 / EXAONE 1 | 49 / 31 | 0.9774 | 0.9467 |
| EXAONE 2 / EXAONE 1 | 77 / 47 | 0.9602 | 0.9564 |

EXAONE의 층 조합은 global attention 범위에서 선택한 탐색 비교다. 실제 초기화 시의 layer 복사 원본을 확정한 것은 아니다.

비교 범위를 확인하도록 다음 표에는 **16개 층 조합 전체의 최소~최대값**을 제시했다. 대응하지 않는 층도 포함하며 최대값은 탐색에서 얻은 값이다. 통계적으로 보정한 판정 점수가 아니다.

| 모델 쌍 | Q UCKA 범위 | K UCKA 범위 |
|---|---:|---:|
| Solar 2 ↔ Solar 1 | 0.0007~0.4984 | 0.0006~0.5580 |
| EXAONE 2 ↔ EXAONE 1 | 0.0123~0.9774 | 0.0224~0.9564 |
| Qwen Instruct ↔ Base | −0.0004~1.0000 | −0.0011~0.9999 |
| Solar 2 ↔ Qwen Base | −0.0005~0.0311 | −0.0006~0.0090 |
| EXAONE 2 ↔ Qwen Base | −0.0005~0.0011 | −0.0004~0.0007 |

같은 모델 family의 알려진 파생 쌍이라도 다른 층을 비교하면 값이 낮을 수 있다. 따라서 “선택한 층에서 낮은 점수가 나옴 → 독립 모델”이라는 판정을 하지 않는다. 이번에는 이전 세대의 선택 대응 층과 외부 비교 후보 사이에서 차이를 관찰했다. 특히 EXAONE의 깊은 선택 층의 높은 상관은 공개된 upcycling 설명과 부합한다.

## GhostSpec 기반 QK 스펙트럼

24개 층의 GQA QK 곱의 0이 아닌 특이값 전체를 수집했다. config 및 tensor shape의 실제 head 크기를 확인하고, [CPU 시범](09-weight-provenance-pilot.md)에 설명한 thin QR 항등식으로 dense 곱을 생략했다.

| 모델 | 선택 층 |
|---|---|
| Solar Open 2 / Solar Open 1 | 각각 0, 4, 20, 44 |
| K-EXAONE 2.0 | 5, 25, 49, 77 |
| K-EXAONE 선행 모델 | 3, 15, 31, 47 |
| Qwen Base / Instruct | 각각 0, 8, 16, 27 |

**이 작업은 지문 수집이며 GhostSpec의 최종 판정 점수가 아니다.** VO 곱, 전체 층 POSA 정렬, effective-rank 기반 최종 scoring은 수행하지 않았다. 특이값 벡터의 모양이 비슷하다는 이유만으로 계승을 판정하지 않는다. AWM 기반 UCKA에서 본 관측과 구분한다.

## 모델별 현재 판단

- **Solar Open 2**: 공개된 선택 가중치 계승 설명과 여러 선택 층·embedding의 결과가 부합한다. 자체 Solar 1에서의 계승 신호이지 외국 모델 차용의 판정이 아니다.
- **K-EXAONE 2.0**: 공개된 upcycling 설명과 여러 global attention 층·embedding의 결과가 부합한다. 실제 초기화 layer 매핑과 선행 모델의 원래 기원은 별도 검증 대상이다.
- **A.X K2**: 정렬 후 새로운 embedding 표본에서도 K1과 낮은 상관을 보였다. K1에서의 embedding 계승 신호를 찾지 못했다는 결과이며, MLA/FP8 attention과 초기 학습 기록이 없으므로 전체 From Scratch 확증은 보류한다.

## 재현과 검증

GPU 런타임은 PyTorch `2.8.0+cu129`, NumPy `2.4.6`였다. TF32를 끄고 float32 행렬 연산을 사용했다. CPU·GPU UCKA 작은 행렬 비교값은 각각 −0.0224367213과 −0.0224367250으로, 차이가 1e−5 미만이었다. GQA compact/dense 특이값 항등식 검사도 통과했다. 기존 수학 검증 3개 역시 클라우드에서 통과했다.

- [GPU 실행 코드](../experiments/llm-dna/scripts/gpu_weight_provenance.py).
- [고정 리비전·표본 token/ID·LAP 대응·전체 층 수치·스펙트럼·읽은 범위/해시](../wiki/raw/2026-10-12_spot-weight-provenance.json).
- [기존 가중치 리더 및 CPU 수학 검증](../experiments/llm-dna/scripts/weight_provenance_probe.py).
- [원 방법론과 접근 조건](08-from-scratch-methods.md).

전체 vocab·전체 layer를 사용하는 원 AWM의 완전 재현이 아니다. 여러 비교에 대한 통계 보정도 수행하지 않았다. 대조군의 제한, sampling, 구조 차이, quantization 및 선택 층의 한계를 함께 읽어야 한다. REEF·LLMPrint·LLM-DNA 추론, HuRef·PoL·PoTD의 개발사 이력 검증은 이번 실행에 포함하지 않았다.

## 비용·자원 종료 확인

승인 조건은 [견적서](10-spot-verification-estimate.md)에 있다. Spot 최고 단가는 $6.50/시간으로 제한했고, AMI 내부의 자동 shutdown과 로컬 외부 종료 감시를 설정했다.

시작부터 첫 종료 상태 관측까지 842.87초를 모두 포함하고 관측 Spot 단가 $5.2599/시간을 적용하면, 인스턴스 비용의 보수적 추산은 **약 $1.23**다. 이 계산에는 청구하지 않는 pending 시간도 포함했다. EBS·공인 IPv4·S3 요청 등 소액 부대비용과 세금은 별도이며, AWS 확정 청구액은 아니다. 실제 총 청구액을 $1.23으로 확정한 것으로 읽으면 안 된다.

인스턴스의 `terminated` 상태, 루트 EBS 삭제, 임시 보안그룹·instance profile·역할·결과 버킷 삭제를 확인했다. 결과를 내려받고 검증한 뒤 임시 자원을 정리했다. [종료·비용 추산·원본 SHA-256 기록](../wiki/raw/2026-10-12_spot-execution.json)에 확인 범위를 보존했다. AWS는 `shutting-down` 또는 `terminated` 상태부터 인스턴스 사용료를 부과하지 않는다. [AWS 상태별 청구 문서](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)
