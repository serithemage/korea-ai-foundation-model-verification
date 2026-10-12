# From Scratch 검증을 확장하는 방법

조사일: 2026-10-12. Exa로 가중치, 내부 표현, 출력 행동, 학습 기록의 4개 관점에서 20개 검색 결과를 검토했다. 중복 및 방법이 맞지 않는 결과를 걸러 핵심 원문 8개를 추가로 읽었다. 아래는 문헌 조사이며 **대상 3개 모델에 이 방법을 실행한 결과가 아니다**.

## 먼저 두 질문을 구분한다

1. **어떤 기존 모델에서 가중치를 이어받았는가?** 최종 모델끼리 비교하는 계통 검사다. 특정 후보의 파생이라는 증거를 찾을 수 있다.
2. **실제로 무작위 초기화에서 학습을 시작했는가?** 학습 과정의 이력 검사다. 초기 상태와 이후 업데이트를 검증해야 한다.

계통 검사에서 후보를 찾지 못해도 두 번째 질문의 답이 자동으로 “예”가 되지는 않는다. 비교하지 않은 부모 모델, 가중치 변환, 긴 추가 학습 등으로 흔적을 놓칠 수 있다. 지식 증류 또한 초기 가중치 재사용과 구분한다. 무작위 초기화 모델이 다른 모델의 출력으로 학습될 수도 있다.

## 공개 가중치로 할 수 있는 검사

| 방법 | 무엇을 비교하나 | 필요한 접근 | 얻는 증거와 한계 |
|---|---|---|---|
| **AWM** | 공통 vocab의 embedding으로 permutation·부호 정렬을 찾은 뒤 Q/K 가중치를 unbiased CKA로 비교 | 양쪽 가중치, vocab, 구조 매핑 | 좌표가 바뀌거나 upcycling을 거친 계승의 탐지 후보. 모든 구조에 자동 적용되지는 않음 |
| **GhostSpec** | attention의 Q–K, V–O 관련 행렬 곱에서 특이값 스펙트럼을 추출하고 layer를 정렬 | 양쪽 attention 가중치, 연산 구조 | 깊이 변경을 고려하는 계통 신호. 스펙트럼만 비슷한 독립 모델을 구분할 대조군 필요 |
| **HuRef** | Transformer 가중치의 변환 불변 항을 지문으로 변환 | 가중치 또는 개발사 측 지문 생성과 검증 가능한 증명 | 가중치를 공개하지 않는 협력적 환경에도 사용 가능. 지문 계산의 증명과 무작위 초기화의 증명은 서로 다름 |
| **REEF** | 같은 입력에 대한 hidden representation을 CKA로 비교 | 양쪽 모델 추론 및 내부 activation 접근 | 출력 문체보다 내부 표현의 유사성을 관찰. 가중치 다운로드만으로 끝나지 않으며 추론 비용 발생 |

출처: [AWM 원문](https://arxiv.org/abs/2510.06738), [GhostSpec 원문](https://arxiv.org/abs/2511.06390), [HuRef, NeurIPS 2024](https://papers.nips.cc/paper_files/paper/2024/file/e46fc33e80e9fa2febcdb058fba4beca-Paper-Conference.pdf), [REEF, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/77e830bd802f4b63b682038d090a97b6-Paper-Conference.pdf).

### AWM: 단순 cosine의 다음 단계

AWM은 공통 토큰의 embedding 열 사이 유사도를 이용해 Linear Assignment Problem을 풀고 permutation과 부호를 복원한다. 그 뒤 Q/K 행렬에 unbiased CKA를 적용한다. 논문은 파생 60쌍과 독립 90쌍으로 구성한 평가에서 강한 분리 결과를 보고한다. 이 성능을 한국 250B~750B MoE에 그대로 적용할 수 있다고 해석하면 안 된다. 논문의 평가 집합과 실제 대상의 구조·dtype·규모가 다르다. [AWM](https://arxiv.org/abs/2510.06738)

우리 적용 판단: **우선 검토할 후보**다. 현재 가중치 표본 검사는 정렬 없이 같은 좌표를 비교했다. AWM을 추가하면 permutation·부호 변환의 영향을 줄이는 방향으로 확장할 수 있다. 다만 필요한 embedding과 attention 행렬을 확보해야 하며, 세 모델의 구조와 FP8 배포본에 맞는 구현 검토가 먼저다.

### GhostSpec: layer 수가 달라진 모델의 비교

GhostSpec은 attention 내부 행렬 곱의 특이값을 지문으로 삼고 POSA라는 layer 정렬 절차를 사용한다. 이는 단순히 모델의 layer 번호를 일대일로 비교하는 방식보다 깊이 변경을 다루는 데 적합한 후보다. [GhostSpec](https://arxiv.org/abs/2511.06390)

우리 적용 판단: EXAONE의 48→78 layer 확장에는 검토 가치가 있다. Solar의 GQA/KDA 혼합, A.X의 MLA/SGA에는 원문의 attention 정의를 그대로 대입하지 않는다. 해당 구조에 대한 수학적 매핑과 FP8 scale 복원 가능성을 먼저 확인한다. 모든 architecture 차이를 해결하는 도구로 소개하지 않는다.

### REEF와 CKA: 내부 표현의 유사성

REEF는 동일 샘플을 입력했을 때 나온 두 모델의 representation을 CKA로 비교한다. 차원 수가 다를 수 있고, 논문은 순열·스케일 변환과 다양한 후속 학습에 대한 실험을 보고한다. [REEF](https://arxiv.org/abs/2410.14273)

우리 적용 판단: LLM-DNA와 다른 관찰 방식으로 교차검증할 수 있다. 두 모델에 같은 입력 집합을 사용하고, layer 선택·token pooling·padding 처리·activation 수집 방식을 고정해야 한다. CKA 유사성은 실제 학습 이력이 아니라 계통의 정황이다. 공유 데이터·공유 작업의 영향을 대조군으로 확인한다.

## 출력만으로 할 수 있는 검사

| 방법 | 핵심 | 필요한 접근 | 한계 |
|---|---|---|---|
| **Model Provenance Testing** | 후보 부모와 대상의 출력 유사도를 관련 없는 모델들과 비교하고 다중 가설검정 | next-token 예측 query, 후보 부모·독립 대조군 | 검정 가정과 대조군 선택에 의존. 일반 채팅 응답만 있는 API와 논문의 query 조건이 다를 수 있음 |
| **LLMPrint** | 모델별 token 선호를 드러내는 fingerprint prompt를 최적화하고 변형 모델에서 유지되는지 검사 | 지문 생성·검증 단계별 query 접근; gray-box/black-box 설정 확인 | post-training·양자화 변형을 다루는 연구. API의 system prompt·필터·버전 변경 영향을 별도로 통제해야 함 |

출처: [Model Provenance Testing, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/30e55dfbfa8e76e3526de7f2ab029cda-Paper-Conference.pdf), [LLMPrint, ACL 2026](https://aclanthology.org/2026.acl-long.541/).

Model Provenance Testing은 기존 위키에 있는 방법의 후속 확인이다. 원문은 30M~4B 모델 600개 이상으로 구성한 두 평가 집합에서 precision 90~95%, recall 80~90%를 보고했다. 이 값은 이번 대상 모델의 판정 신뢰도가 아니다. 사용자가 정한 오탐 수준을 논문의 가정 아래 통제하는 검정이며, 무관 모델의 올바른 구성과 여러 후보 검사에 대한 보정이 필요하다. [원문](https://proceedings.neurips.cc/paper_files/paper/2025/file/30e55dfbfa8e76e3526de7f2ab029cda-Paper-Conference.pdf)

LLMPrint는 사전 워터마크 없이 이미 공개된 모델에서 특화된 prompt 지문을 구성하는 방향이다. 논문은 5개 base 모델과 약 700개 후속 학습·양자화 변형을 평가했다. 이것도 한국 대형 MoE에서의 성공을 보장하지 않는다. [원문 및 코드 안내](https://aclanthology.org/2026.acl-long.541/)

워터마크나 backdoor는 개발사가 미리 넣은 신호가 있을 때 소유·계승 검증에 활용할 수 있다. 이미 공개된 모델에 사후로 없는 워터마크를 검사할 수는 없다. 또한 워터마크의 존재 자체가 From Scratch를 증명하지 않는다. 기존 모델에도 새 신호를 넣을 수 있기 때문이다. [REEF의 관련 연구 검토](https://arxiv.org/abs/2410.14273)

## 개발사의 학습 기록이 있어야 할 수 있는 검사

| 방법 | 핵심 | 필요한 자료 | From Scratch 확증과의 관계 |
|---|---|---|---|
| **Proof-of-Learning (PoL)** | checkpoint 사이 업데이트를 재실행해 기록된 학습 경로가 맞는지 검사 | 초기·중간·최종 가중치, 학습 설정, 데이터 순서 등 | 초기 상태 검증과 결합해야 함. 일부 경로의 replay만으로 전체 무작위 초기화를 증명하지 못함 |
| **Proof-of-Training-Data (PoTD)** | seed 사전 commitment와 학습 transcript 검증 등을 통해 학습 데이터·과정 주장을 확인 | seed commitment, 데이터, 코드, 중간 checkpoint, 개발사 협조 | 학습 기록의 위조 위험을 다루는 접근. 무엇을 commitment하고 검증했는지 명시해야 함 |

출처: [Proof-of-Learning: Definitions and Practice](https://arxiv.org/abs/2103.05633), [Tools for Verifying Neural Models’ Training Data, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/03e33e1f62e3302b47fe1d38a235921e-Paper-Conference.pdf).

단순 로그 캡처보다 더 강한 증거를 얻으려면, 학습 전에 초기화 seed·코드·설정과 초기 checkpoint의 commitment를 남기고 이후 기록을 연결해야 한다. 일부 구간을 verifier가 재실행하고 최종 공개본까지 연결되는지 확인하는 방식이 후속 설계 후보다. 이는 두 논문을 참고한 **우리의 적용 제안**이며 논문이 이번 대형 MoE의 전체 학습을 이미 증명했다는 뜻이 아니다. 분산 학습·하드웨어 차이에 따른 재현 허용오차, 데이터 제공, 기록 저장 비용도 확인해야 한다.

모든 초기 학습 파라미터가 주장한 초기화 코드에서 생성됐는지도 별도로 확인해야 한다. seed 공개나 초기 checkpoint 한 개만으로는 그 뒤 몰래 다른 checkpoint를 주입하지 않았다는 사실까지 보장하지 못한다. 일부 구간 표본 검증에도 미검사 구간의 한계가 남는다.

PoTD 원문은 기존 PoL 공격을 논의하고 추가 검사법을 제안한다. 따라서 “학습 로그가 있으면 증명 완료”라고 처리하지 않는다. 아울러 추론 계산을 검증하는 zero-knowledge proof와 전체 학습 경로를 검증하는 proof는 구분한다. 이번 조사에서는 대형 MoE의 완전한 학습 증명을 즉시 실행할 수 있는 도구를 확인한 것으로 보고하지 않는다.

## 이 프로젝트에 적용할 순서

아래 순서는 문헌과 이번 공개본 검사의 한계를 바탕으로 한 제안이다. **추가 GPU 작업이나 모델 추론은 아직 실행하지 않았다.**

1. **평가 제출본의 식별부터 고정한다.** 실제 제출 checkpoint와 공개 저장소 리비전을 연결할 수 있어야 같은 모델을 검사한다.
2. **알려진 계승을 대조군으로 만든다.** Solar 1→2, EXAONE 1→2를 공개된 계승 사례로 사용하되, 보정에 쓴 쌍을 최종 성능 평가에 다시 사용하지 않는다. 추가로 계통이 문서화된 파생 모델과 독립 모델을 확보한다.
3. **가중치 기반 AWM·GhostSpec의 구조 호환성을 검토한다.** 필요한 행렬·dtype를 조사하고, FP8 scale·layer/expert 매핑을 정의한다. 전체 다운로드나 계산을 시작하기 전에 실제 크기에 따른 비용을 산정한다.
4. **가능한 경우 REEF·출력 가설검정으로 교차검증한다.** 알려진 부모만을 비교하지 말고 독립 대조군과 후보 부모 집합을 포함한다. prompt·tokenization·추론 설정을 고정하고 여러 후보 검사에 맞는 오탐 보정을 한다.
5. **확정 판정에는 초기화·학습 transcript를 요청한다.** 공개본 비교가 주는 계통 증거와 개발사가 제공하는 학습 이력 증거를 합쳐 판단한다.

결과 보고는 “후보 A에서의 계승 신호를 찾음”, “선택한 검사에서 계승 신호를 찾지 못함”, “검증 가능한 무작위 초기화·학습 이력을 확인함”을 구분한다. 어느 경우든 확인 범위 밖의 보편적 결론으로 확대하지 않는다.

## 이번 검색에서 핵심 방법으로 채택하지 않은 결과

- **Neural Lineage (CVPR 2024)**: 흥미로운 일반 신경망 계통 연구지만 이번 대형 LLM 대상에 대한 직접 근거로 대입하지 않았다.
- **zkLLM**: 추론의 진위 검증과 전체 학습·초기화 검증을 혼동하지 않기 위해 분리했다.
- **이미지 분류 모델의 adversarial fingerprint**: LLM에서 같은 성능을 보장하는 근거로 사용하지 않았다.
- **Stemma 등 추가 후보**: 이번 범위에서는 위의 8개 핵심 원문을 우선 검토했다. 목록이 관련 연구 전체를 망라하지는 않는다.
