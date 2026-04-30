# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

**국가 AI 파운데이션 모델 "From Scratch" 검증 프로젝트** — 5개 기관(Upstage, NAVER, SKT, NC AI, LG AI)의 공개 모델이 from scratch로 학습되었는지 검증하는 문서/분석 저장소.

코드 빌드/테스트 시스템은 없으며, 산출물은 모두 마크다운 문서다. 검증 대상 모델 목록과 판정 결과는 [README.md](README.md)의 "검증 대상 모델" 표를 단일 출처로 사용한다 — 여기서 중복하여 적지 않는다.

## 사용 가능한 커맨드

| 커맨드 | 설명 |
|--------|------|
| `/save [메시지]` | 변경사항 분석 후 커밋 & 푸시 (메시지 생략 시 자동 생성) |
| `/update-tutorial` | 현재 대화의 Q&A를 `docs/tutorial/` 주제별 파일에 추가 |
| `/update-changelog` | 현재 대화의 변경 사항을 `CHANGELOG.md`에 추가 |

커맨드 정의는 `.claude/commands/`, 스킬은 `.claude/skills/` 참조.

## 핵심 워크플로우: 질문 vs 명령

사용자 입력을 **질문**과 **명령**으로 구분하여 처리한다. 질문이면 답변 후 **반드시** `docs/tutorial/`에 Q&A를 추가한다 (외부 조사 없이 답할 수 있어도 마찬가지).

### 질문으로 판별하는 패턴

- `?`로 끝남 / `설명해`, `알려줘` / `~인가요`, `~인지` / `뭐야`, `뭔가요` / `어떻게`, `왜`
- 개념·용어 질문 ("토큰 중복률이란", "BPE Merge Rules란")

### 튜토리얼 업데이트 예외 (질문이어도 추가 안 함)

- 프로젝트 운영 질문 ("파일 어디 있어?", "커밋 어떻게 해?")
- Claude 사용법 질문
- 단순 확인 질문 ("이거 맞아?", "진행해도 돼?")

### Perplexity MCP 사용 규칙

외부 조사가 필요한 질문은 Perplexity MCP를 사용한다.

1. 쿼리는 **영문**으로 작성 (글로벌 CLAUDE.md 지침)
2. 결과는 **한국어**로 정리
3. 주제에 맞는 `docs/tutorial/*.md` 파일의 `<!-- SECTION_MARKER -->` **위에** Q&A 삽입

### Q&A 주제별 분류

| 주제 | 대상 파일 |
|------|----------|
| from scratch 개념, 검증 방법 개요 | `docs/tutorial/01-기초개념.md` |
| Tokenizer, vocabulary, BPE, merge rules | `docs/tutorial/02-tokenizer-분석.md` |
| Weight, cosine similarity, architecture | `docs/tutorial/03-weight-architecture-분석.md` |
| 특정 모델 검증 결과, 논란 사례 | `docs/tutorial/04-사례연구.md` |
| 방법론 비판, 학술 연구, 개선 방향 | `docs/tutorial/05-방법론-평가.md` |

신규 Q&A 작성 시 주제 분류, Q 번호 규칙, 내러티브 스타일 가이드는 `.claude/skills/update-tutorial/SKILL.md`(스타일 상세는 `.claude/skills/update-tutorial/references/narrative-style.md`)를 따른다.

### Q&A 형식 (내러티브 스타일)

```markdown
---

## Q{N}: {질문 요약}

**질문 시각**: {YYYY-MM-DD}

**답변**:

{내러티브 형식 — 이야기를 풀어가듯 서술. 테이블/bullet 나열 지양}

---
```

**작성 원칙**: 인과관계 설명, 비교/대조 활용, 발견 과정 서술. 테이블은 핵심 수치 요약 보조 수단으로만 사용.

## 문서 구조

```
docs/
├── tutorial/                       # Q&A 학습 기록 (사용자 질문 → 여기에 누적)
│   ├── README.md                   # 인덱스 + 목차
│   └── 01..05-*.md                 # 주제별 Q&A
└── 0N-*-analysis.md                # 검증 방법론 + "모델별 검증 결과" 섹션
```

- `docs/0N-*-analysis.md` (4종, tokenizer/weight/architecture/behavior): 방법론 설명과 함께 5개 모델의 분석 결과를 누적 기록한다.
- 새 모델 검증 시: 각 분석 문서의 **"모델별 검증 결과"** 섹션에 추가 → README.md 검증 대상 표 상태 업데이트.

## 주요 파일

| 파일 | 역할 |
|------|------|
| `README.md` | 메인 문서, 검증 대상 모델 표 (단일 출처) |
| `CHANGELOG.md` | 변경 이력 (날짜별, 카테고리별) |
| `docs/tutorial/README.md` | Q&A 튜토리얼 인덱스 |
| `.claude/commands/save.md` | `/save` 커맨드 정의 |
| `.claude/skills/update-tutorial/SKILL.md` | Q&A 작성 스타일 가이드 (내러티브) |
| `.claude/skills/update-changelog/SKILL.md` | CHANGELOG 작성 가이드 |
| `.claude/skills/llm-dna-knowledge/SKILL.md` | LLM-DNA·계통 분석 작업 시 wiki 우선 참조 컨벤션 |
| `.claude/skills/llm-dna-extraction-playbook/SKILL.md` | LLM-DNA SageMaker spot 잡 운영 9가지 교훈 (Phase 4–6 회고 기반, $34 학습 곡선 비용 사전 차단용) |
| `.claude/skills/karpathy-guidelines/SKILL.md` | LLM 코딩 함정 회피 4원칙 (Think/Simplicity/Surgical/Goal-Driven) — [forrestchang/andrej-karpathy-skills](https://github.com/forrestchang/andrej-karpathy-skills), MIT |
| `wiki/index.md` | LLM-DNA 지식 베이스 인덱스 (28페이지, [[wiki-link]] 교차참조) |
| `experiments/llm-dna/` | LLM-DNA 추출 실험 (CDK + SageMaker spot training) |

## LLM Wiki 활용 규칙

이 프로젝트는 `wiki/` 디렉토리에 `[[wiki-link]]` 교차참조 위키를 유지한다. Karpathy LLM Wiki 패턴을 따른다.

### 작업 시 우선순위

1. **위키 우선 참조**: 질문에 답하거나 작업을 시작할 때, `wiki/index.md`에서 관련 페이지를 먼저 찾아 읽는다. `[[wiki-link]]`를 따라 2-hop까지 확장하여 맥락을 파악한다.
2. **출처 인용**: 답변 작성 시 위키 페이지를 `[[페이지 제목]]` 형태로 인용한다.
3. **위키에 없으면 명시**: 위키에 답이 없으면 "위키에 없음"이라 표기하고 `wiki/raw/` 또는 외부 원본을 읽어 보강한다. 새 내용이면 `llm-wiki ingest`로 위키를 성장시킨다.
4. **검색보다 컴파일**: 매번 원본을 다시 읽지 말고, 이미 컴파일된 위키 지식을 적극 재활용한다.

### 위키 구조

- `wiki/index.md` — 자동 재빌드되는 라우팅 레이어 (수동 편집 금지)
- `wiki/{page}.md` — 컴파일된 지식 페이지 (`[[wiki-link]]` 교차참조)
- `wiki/raw/` — 원본 소스 드롭존 (불변)
- `wiki/log.md` — ingest 이력
- `wiki/.lancedb/` — 선택적 벡터 인덱스 (lancedb-sync 실행 시)

### 스킬 명령

사용자가 `llm-wiki <command>`를 호출하면 해당 단계 실행:
- `init` — 구조 초기화
- `ingest <source>` — 새 소스를 컴파일해 위키 갱신
- `query <질문>` — 하이브리드 검색(qmd) 또는 INDEX.md 라우팅
- `lint` — 끊어진 링크·고아 페이지 탐지
- `sync` — index.md 재빌드
- `export <template>` — 온보딩·ADR 요약 등 출력
- `qmd-index` — qmd 인덱스 빌드 (선택)
- `lancedb-sync` — LanceDB 벡터 인덱스 동기화 (선택)

## 인프라 운영 규칙 (LLM-DNA Phase 4–6 회고 기반)

`experiments/llm-dna/` 또는 SageMaker training job을 다루는 모든 작업은 아래 9개 규칙을 따른다. 정량 근거와 사례는 [docs/06-cost-and-architecture-lessons.md](docs/06-cost-and-architecture-lessons.md)를, 자동 활성화 체크리스트는 `.claude/skills/llm-dna-extraction-playbook/SKILL.md`를 참조한다. 이 규칙들은 한 라운드($34 학습 곡선 비용)를 거쳐 도출된 *deterministic 안전 장치*이며 새 모델 검증·재시도 시 반드시 통과해야 한다.

### 잡 제출 전 게이트

1. **Spot quota 사전 확인**: SageMaker spot quota는 EC2 placement score와 독립이다. score=9여도 quota=0이면 못 쓴다. `aws service-quotas list-service-quotas --service-code sagemaker --region <r>`로 *제출 전* 확인하고, 부족 시 다른 리전으로 라우팅한다 — `submit_dna.py --region` 인자가 그 용도.

2. **GPU 호환성 매트릭스 검증**: MoE 모델(Mixtral, Solar-Open, EXAONE MoE 계열)은 transformers 5.x의 `torch._grouped_mm` 사용으로 인해 Hopper(CC 9.0, H100/H200) 전용이다. L40S(8.9), A100(8.0), A10G(8.6)에서는 RuntimeError. yaml의 `instance_type`을 모델 architecture에 맞춰 명시해야 하며, `min_cuda_capability` 필드를 추가하면 자동 검증 가능.

3. **Sanity gate를 entrypoint에 통합**: `dna_train.py`는 모델 로드 직후 1-prompt forward 검증을 *반드시* 수행하고 응답 길이 < 20이면 non-zero exit. 이 한 게이트가 빈 응답 silent failure, weight conversion 실패, GPU 호환성 RuntimeError를 모두 1분 안에 잡는다.

### 잡 실행 중 거버넌스

4. **30분 이상 wall time 잡은 중간 폴링 의무**: cold start + weights load 후 generation 단계의 `s/prompt`를 첫 5–10 prompt에서 측정해 expected ceiling을 넘으면 즉시 stop. Solar 1차 잡(200s/prompt)에서 25분 만에 잡은 사례가 그렇지 않았으면 $135까지 갈 비용을 $7.93에 멈췄다.

5. **`max_new_tokens=256` cap을 default로**: sentence-encoder(`all-mpnet-base-v2`)의 ~512 토큰 컷오프 때문에 cap=256은 *fingerprint 무손실*. cap 없으면 base 모델이 max=2048까지 흘러 generation 시간 8× 증가. yaml `dna_extraction.max_new_tokens=256`이 default여야 하고 entrypoint에 `patch_model_wrapper_for_min_new_tokens(max_cap=256)`이 적용된다.

6. **Deterministic 실패는 2회 후 격리**: 같은 에러 로그가 환경 차이 없이 두 번 반복되면 *재시도 금지*. K-EXAONE-236B 8회 재시도 ($13)가 이 규칙 부재의 비용이었다. 격리 옵션: (a) 모델 제외 + 사유 명시, (b) 같은 가족의 대체 모델로 substitute(K-EXAONE-236B → EXAONE-3.5), (c) 상류 라이브러리 호환성 회복 시점까지 deferral.

### 잡 종료 후 검증

7. **`--continue-on-error`는 통과 임계값과 페어링**: 모든 prompt가 실패해도 잡이 *Completed*로 보고되는 양면성이 있다. entrypoint 끝에서 `responses.json`의 nonempty ratio < 0.9이면 non-zero exit으로 *Failed* 마킹 → spot 자동 재시도까지 받는다.

8. **결과 매칭은 latest-wins**: S3는 append-only이고 잡 prefix는 ISO timestamp(`YYYYMMDD-HHMMSS`)를 포함하므로 `max(prefix)` = chronologically latest. `analyze.py` 같은 후속 스크립트는 model_id별로 latest를 자동 선택해야 옛 빈 응답 결과가 정상 결과를 가리는 silent bug를 막는다.

### 인프라 일반

9. **HF cache는 `/tmp` (instance store SSD)로 redirect**: SageMaker 컨테이너의 root 볼륨(120GB)이 아니라 NVMe instance store(g7e.48xl 1.7TB, p5.48xl 28TB+)를 사용해야 100B+ 모델이 들어간다. `dna_train.py`의 `os.environ.setdefault("HF_HOME", "/tmp/.cache/huggingface")` + `HF_HUB_DISABLE_XET=1` + `HF_HUB_ENABLE_HF_TRANSFER=0` 조합이 검증된 설정.

### 멀티 리전 stack

CDK는 us-east-1 + us-west-2 동시 배포가 default다(`cdk/bin/llm-dna.ts`의 `CDK_TARGET_REGION` 환경변수, us-east-1은 짧은 stack name `LlmDnaStack` 유지, 다른 리전은 `LlmDnaStack-<region>` 접미). `roleName` 같은 IAM 글로벌 unique 자원은 제거해 충돌을 방지했다. 새 리전 추가 시 `cdk deploy --region <r>` 후 `submit_dna.py --region <r>`로 라우팅한다.

### 비용 통제

Spot 사용은 협상 불가능한 default다(on-demand 대비 ~68% 절감). 누적 비용 추적은 `aws sagemaker list-training-jobs` + `BillableTimeInSeconds`로 정량 가능하며, 라운드 종료 시 `docs/06-cost-and-architecture-lessons.md` 같은 회고 문서에 *낭비 카테고리*를 분리 기록해 다음 라운드에 반영한다.
