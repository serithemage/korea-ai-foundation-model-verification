# 2차평가 진출 모델의 From Scratch 예비검증

검증일: 2026-10-12. 대상은 진출 3개사의 **공식 공개본**이다. 실제 평가 제출 체크포인트와 공개본의 동일성은 확인하지 못했다. 이번 검증은 공개 자료, 토크나이저, 구조, 선택 가중치 표본을 다룬다.

## 현재 판정

이 보고서에서 **세대별 From Scratch**는 해당 모델의 학습 가중치를 모두 무작위로 초기화해 사전학습한 경우를 뜻한다. 토크나이저나 공개 알고리즘의 재사용은 별도로 기록한다. **자체 계통 계승**은 개발사의 선행 모델 가중치를 사용하는 경우다. 선행 모델의 기원까지 독립 검증했다는 뜻은 아니다. 정부의 사업 적격성 판정은 이번 검증 범위에 포함하지 않는다.

| 모델 | 개발사가 공개한 학습 경로 | 세대별 From Scratch 판정 | 자체 계통·외부 차용 판정 |
|---|---|---|---|
| Solar Open 2 | Solar Open 1 가중치 5.69B, 약 2.3%를 선택적으로 이전 | **전체 무작위 초기화에 해당하지 않음** — 개발사 보고 기준 | 자체 선행 모델에서 계승. 선행 모델 기원과 외부 가중치 차용 여부는 미확정 |
| K-EXAONE 2.0 | K-EXAONE을 깊이·expert 수 방향으로 확장한 뒤 계속 사전학습 | **전체 무작위 초기화에 해당하지 않음** — 개발사 보고 기준 | 자체 선행 모델에서 계승. 선행 모델 기원과 외부 가중치 차용 여부는 미확정 |
| A.X K2 | From Scratch 사전학습을 명시 | **개발사 주장과 관측 표본은 부합하나 독립 확증은 보류** | K1 토크나이저 재사용 확인. 전체 가중치 기원은 미확정 |

출처: [Solar Open 2 보고서 §3.1](https://arxiv.org/html/2607.20062v2), [K-EXAONE 2.0 보고서 §2.2](https://arxiv.org/html/2608.04505), [A.X K2 보고서](https://arxiv.org/html/2608.30181). 앞의 두 모델을 외국 모델의 파인튜닝으로 판정한 것이 아니다.

## 대상과 증거 고정

[2026-08-18 연합뉴스 기사](https://www.yna.co.kr/view/AKR20260818120600017)는 LG AI연구원, SK텔레콤, 업스테이지의 다음 단계 진출과 모델명을 보도했다. 이번에는 해당 모델의 원 개발사 저장소를 사용했다. 파생 양자화 저장소나 다른 크기의 대체 모델은 사용하지 않았다.

| 역할 | Hugging Face 저장소 | 고정 리비전 |
|---|---|---|
| Solar 2 | `upstage/Solar-Open2-250B` | `9190fbe63a2ad8e17fc766ccceb36de7c66f004b` |
| Solar 1 비교본 | `upstage/Solar-Open-100B` | `1f591439b14055004d1a5d1a975608953a022fea` |
| A.X K2 | `skt/A.X-K2` | `2287ca456927eed33b899c404c0b2ffaf17aa09f` |
| A.X K1 비교본 | `skt/A.X-K1` | `bb56f50f36b6c0b7b0c108da02db47a5e3bd9ee1` |
| EXAONE 2 | `LGAI-EXAONE/K-EXAONE-2.0-750B-A37B` | `fec0c8dd25fdd20ffd09869d5af5277966d5523d` |
| EXAONE 1 비교본 | `LGAI-EXAONE/K-EXAONE-236B-A23B` | `61e6d578eb102b578e5704e2916ac841df9eca0a` |

[증거 JSON](../wiki/raw/2026-10-12_second-round-evidence.json)에 config 원문, 파일 SHA-256, 비교 지표, shard 이름, HTTP byte range, 가중치 표본 원본을 보존했다. 토크나이저와 전체 weight index 원문은 고정 리비전 URL로 다시 받을 수 있다. 표본 해시는 전체 tensor 해시가 아니다.

## 구조와 배포 dtype 실측

다음 값은 각 리비전의 `config.json`에서 읽었다. 파라미터 규모는 보고서의 표기이므로 저장된 파일의 tensor 개수와 구분한다.

| 항목 | Solar Open 2 | A.X K2 | K-EXAONE 2.0 |
|---|---:|---:|---:|
| 보고서 총/활성 규모 | 250B / 15B | 688B / 33B | 750B / 37B |
| hidden size | 4,096 | 7,168 | 6,144 |
| layers | 48 | 61 | 78 |
| routed experts | 320 | 256 | 256 |
| experts per token | 8 | 8 | 8 |
| config vocab size | 196,608 | 163,840 | 153,600 |
| config 최대 context | 1,048,576 | 262,144 | 262,144 |
| 배포 dtype | BF16 | FP8 설정, 선택 embedding/head는 BF16 | BF16 |

A.X K2는 config의 `dtype=bfloat16`과 별도로 `quantization_config.quant_method=fp8`을 포함한다. 실제 선택 attention tensor도 `F8_E4M3`였다. config dtype만 보고 모든 가중치를 BF16으로 취급하면 안 된다. 구조 차이는 가중치 이전을 배제하지 않는다.

## 토크나이저 전체 비교

`tokenizer.json`의 `model.vocab`과 전체 BPE merges를 비교했다. 문자열 형식과 2원소 배열 형식의 merge는 동일한 pair 표현으로 정규화했다. Jaccard는 토큰 문자열 집합의 교집합/합집합이며, 동일 ID 개수는 같은 문자열이 같은 ID를 가진 경우만 센다.

| 이번 세대 ↔ 이전 세대 | vocab Jaccard | 동일 문자열·ID 개수 | 전체 merge 순서 일치 |
|---|---:|---:|---:|
| Solar 2 ↔ Solar 1 | 99.9542% | 196,560 / 196,608 | 192,256 / 192,256 |
| A.X K2 ↔ K1 | 100% | 163,840 / 163,840 | 163,399 / 163,399 |
| EXAONE 2 ↔ EXAONE 1 | 99.9766% | 153,327 / 153,600 | 152,982 / 152,982 |

Solar는 이전 vocab 대비 문자열 45개를 교체했고 공통 문자열 3개의 ID가 달랐다. EXAONE은 문자열 18개를 교체했고 공통 문자열 255개의 ID가 달랐다. 따라서 BPE 규칙 계승과 파일 전체 동일성을 구분해야 한다. A.X K2와 K1의 tokenizer 파일 SHA-256은 동일했다.

외부 비교본으로 `Qwen/Qwen3.5-397B-A17B`와 `zai-org/GLM-5`도 리비전을 고정했다. 두 비교본의 ID를 포함한 vocab 또는 전체 merge 목록과 완전히 같은 대상은 없었다.

| 대상 | Qwen3.5 vocab Jaccard | GLM-5 vocab Jaccard | Qwen / GLM의 같은 위치 merge 일치 개수 |
|---|---:|---:|---:|
| Solar 2 | 26.7949% | 29.7817% | 8 / 0 |
| A.X K2 | 25.9469% | 31.7027% | 4 / 2 |
| EXAONE 2 | 28.0385% | 31.2213% | 3 / 1 |

이 수치는 선택 비교본과의 **토크나이저 동일성 검사**다. 외부 가중치 차용 여부를 판정하는 임계값으로 사용하지 않는다. Kimi-K2-Instruct는 `tokenizer.json` 대신 `tiktoken.model`을 배포하므로 이번 JSON 기반 비교에서 제외했다. 모델이 비공개이거나 존재하지 않는다는 뜻은 아니다.

## 선택 가중치 직접 비교

### 표본 추출 방법

모델 전체를 로드하지 않고 safetensors 헤더와 필요한 byte range만 받았다. HTTP 206, `Content-Range`, 응답 길이를 확인했다. BF16 값을 little-endian uint16으로 읽고 상위 16비트에 배치해 FP32 값으로 복원했다. 원격 모델 코드는 실행하지 않았다.

각 tensor를 저장 순서로 펼쳤을 때 원소 수를 N이라 하고, **0, floor((N−2048)/2), N−2048**에서 각각 2,048개를 읽었다. 선택 위치는 결과를 보기 전에 정했다. embedding과 output head의 표본에 걸친 token ID는 양쪽에서 같은 토큰 문자열임을 확인했다.

- `model.embed_tokens.weight`, `lm_head.weight`: 세 쌍 모두 비교.
- `model.layers.0.self_attn.q_proj.weight`: Solar와 EXAONE의 같은 이름·shape tensor를 탐색적으로 비교. EXAONE upcycling의 실제 layer 매핑을 복원한 검사는 아니다.
- `model.layers.0.self_attn.q_a_proj.weight`: K2가 FP8이라 K1 BF16과의 수치 비교는 제외. scale 적용 없이 FP8 byte를 BF16으로 해석하지 않았다.

총 24개 대응 window를 비교했고, 대응하지 않는 K1 attention 표본까지 포함해 읽은 가중치 원본은 208,896 bytes다. 이는 대형 모델 전체의 극히 작은 일부다.

### 결과

표의 세 값은 시작·중간·끝 순서의 **centered cosine**이다. 각 window의 평균을 별도로 뺀 뒤 dot product를 양쪽 L2 norm의 곱으로 나눴다.

| 대상 ↔ 이전 세대 | embedding | output head | 선택 attention projection |
|---|---|---|---|
| Solar 2 ↔ Solar 1 | 0.4626 / 0.5317 / 0.2751 | 0.6233 / 0.8231 / 0.5771 | 0.1476 / 0.1023 / 0.1578 |
| A.X K2 ↔ K1 | 0.0004 / 0.0047 / 0.0424 | −0.0097 / −0.0526 / 0.0477 | 미비교: FP8 |
| EXAONE 2 ↔ EXAONE 1 | 0.5726 / 0.6384 / 0.7078 | 0.8845 / 0.6910 / 0.8550 | 0.7103 / 0.8289 / 0.8002 |

비교한 24개 대응 window 중 원본 byte가 완전히 같은 window는 없었다. 추가 학습 후에는 이전한 가중치도 달라질 수 있으므로 해시 불일치는 재사용을 배제하지 않는다.

간단한 대조로 대상 window를 이전 세대의 다음 위치 window와 순환 비교했다. 대응 비교 24건에서 이 대조의 raw cosine 범위는 −0.0527~0.0523이었다. Solar·EXAONE의 embedding/head 대응 표본은 이 대조보다 높고, 개발사가 밝힌 계승 경로와 부합한다. 이 대조는 무작위 초기화 모델 집단으로 보정한 통계 검정이 아니다. p-value나 판정 임계값을 제시하지 않는다.

A.X K2의 낮은 표본 상관은 **선택한 좌표에서 K1 가중치가 그대로 남아 있다는 증거를 찾지 못했다**는 뜻이다. 무작위 초기화의 증명이나 모든 외부 모델에 대한 차용 배제는 아니다. 회전, permutation, 긴 추가 학습, 비교하지 않은 tensor의 재사용은 탐지하지 못할 수 있다.

### 재현 절차

1. 증거 JSON의 `models`에서 저장소·리비전·파일 SHA-256을 읽는다. `https://huggingface.co/{repo}/resolve/{revision}/{filename}`에서 파일을 받고 해시를 확인한다.
2. `model.safetensors.index.json`의 weight map으로 shard를 찾는다. shard의 처음 8 bytes를 little-endian uint64로 해석해 헤더 길이 H를 얻고, 이어지는 H bytes를 JSON으로 읽는다.
3. tensor의 `data_offsets[0]`을 D, 표본 원소 offset을 i라 하면 BF16 표본의 시작 byte는 `8 + H + D + 2*i`다. 4,096 bytes를 요청한다. 증거 JSON의 `byte_range` 및 `sha256`과 대조한다.
4. 저장된 `raw_bf16_base64`만으로도 수치 계산을 재현할 수 있다. 복원 값에 위 centered cosine을 적용한다. `weight_comparisons`에는 raw cosine, centered cosine, 상대 L2, 동일 원소 개수, 대조 값이 함께 있다.

## 이번 검증의 한계와 후속 조건

기존 1차평가 분석의 “architecture 불일치 → From Scratch 지지”나 “tokenizer 불일치 → From Scratch” 해석을 이번 판정에 적용하지 않았다. 구조를 바꿔도 일부 가중치를 이전할 수 있고, 동일 토크나이저로 새 가중치를 학습할 수도 있다. 이번 두 계승 사례와 A.X 토크나이저 결과가 그 구분의 필요성을 보여준다.

확정하지 못한 항목은 다음과 같다.

- 실제 평가 제출본의 commit/checkpoint 해시와 공개본의 동일성.
- 세 모델의 초기 체크포인트, seed, 학습 로그에 대한 독립 검사.
- Solar 1·K-EXAONE 1의 원래 가중치 기원. 기존 프로젝트의 신뢰 판정을 독립 확증으로 승계하지 않는다.
- 외부 모델의 전체 가중치와의 token 정렬, layer/expert 매핑, permutation·회전 보정 비교.
- FP8 attention의 scale을 포함한 복원, 무작위 모델 및 알려진 파생 모델 대조군을 통한 통계 보정.
- LLM-DNA/행동 분석. 이번에는 실행하지 않았으며 기존 세대나 다른 크기의 결과를 대입하지 않았다.

따라서 현재의 완료 범위는 **공개 자료 검토·전체 토크나이저 비교·제한된 가중치 표본 검사**다. From Scratch의 독립 확증은 계속 보류한다.
