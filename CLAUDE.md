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

신규 Q&A 작성 시 주제 분류, Q 번호 규칙, 내러티브 스타일 가이드는 `.claude/skills/update-tutorial.md`를 따른다.

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
| `.claude/skills/update-tutorial.md` | Q&A 작성 스타일 가이드 (내러티브) |
| `.claude/skills/update-changelog.md` | CHANGELOG 작성 가이드 |
| `.claude/skills/llm-dna-knowledge.md` | LLM-DNA·계통 분석 작업 시 wiki 우선 참조 컨벤션 |
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
