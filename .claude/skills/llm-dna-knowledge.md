# LLM-DNA Knowledge (프로젝트 로컬)

**Purpose**: 본 프로젝트의 `wiki/` 지식 기반(28페이지)을 LLM lineage·model verification 작업에서 항시 참조 가능하게 하는 컨벤션 정의.

이 스킬은 `~/.claude/skills/llm-dna-knowledge/` 글로벌 스킬의 일반 가이드를 본 프로젝트에 맞춰 구체화한다.

## 트리거 조건

다음 중 하나라도 해당하면 무조건 wiki/index.md를 먼저 읽는다:

- LLM-DNA / 모델 계통 / 혈통 / phylogenetic / neighbor-joining 관련 질문 또는 작업
- 검증 대상 3종 작업: [[solar-open-100b]], [[k-exaone-236b]], [[ax-k1]]
- 검증 방법론: tokenizer / weight / architecture / behavior 분석
- 독파모 사업, Phase 평가, from scratch 논란
- `experiments/llm-dna/` 디렉토리 작업
- Reference 모델 비교 ([[llama-3-family]], [[qwen-25-family]], [[glm-family]], [[mixtral-8x7b]], [[deepseek-v25]])

## 본 프로젝트 wiki 구조 (28페이지)

| 카테고리 | 페이지 |
|---------|--------|
| **LLM-DNA 핵심** (5) | llm-dna-overview, dna-extraction-pipeline, random-projection, llm-dna-package, inheritance-and-determinism |
| **계통 분석 알고리즘** (4) | phylogenetic-tree, neighbor-joining, upgma, distance-metrics |
| **검증 대상 3종** (3) | solar-open-100b, k-exaone-236b, ax-k1 |
| **Reference 모델군** (5) | llama-3-family, qwen-25-family, glm-family, mixtral-8x7b, deepseek-v25 |
| **검증 방법론** (6) | tokenizer-analysis, weight-analysis, architecture-analysis, behavior-analysis, model-provenance-testing, layernorm-fingerprint-fallacy |
| **인프라 패턴** (3) | sagemaker-spot-training, aws-cdk-typescript, huggingface-hub-usage |
| **정책·맥락** (2) | dokpamo-project, from-scratch-debate |

전체 인덱스: `wiki/index.md` (`sync_index.py`로 자동 재빌드).

## 작업 절차

### 1. wiki/index.md 먼저 읽기

```
Read wiki/index.md
```

`type` 그룹별로 28개 페이지가 정리돼 있다. 키워드와 페이지 제목을 매칭해 후보 3~5개 식별.

### 2. 후보 페이지 + [[link]] 2-hop 확장

각 후보 페이지를 읽고, 안의 `[[wiki-link]]`를 따라 1~2단계까지 확장. 예시:

- "Solar의 from scratch 신뢰 가능?" → `solar-open-100b.md` → `[[from-scratch-debate]]` + `[[layernorm-fingerprint-fallacy]]` + `[[glm-family]]` → 이들도 읽음

### 3. 답변에 `[[wiki-link]]` 인용

답변에 위키 페이지를 인용할 때 `[[페이지 제목]]` 형식. 외부 자료는 일반 마크다운 링크.

### 4. 위키 갱신 트리거

다음 경우 사용자에게 "위키 업데이트할까요?" 제안:

- 새로운 검증 결과가 도출됐을 때 (예: LLM-DNA 분석 완료 후 결론)
- 새로운 reference 모델이 추가됐을 때
- from-scratch 정의 등 정책 업데이트가 발생했을 때
- 사용자가 새 질문에 외부 검색이 필요할 때 (raw에 드롭 후 ingest)

업데이트 절차는 `documentation:llm-wiki` 스킬의 `ingest` / `lint` / `sync` 명령 참조.

## 본 프로젝트 특수 컨벤션

### Q&A vs Wiki

본 프로젝트는 **Q&A 튜토리얼**(`docs/tutorial/`)과 **위키**(`wiki/`)를 동시에 운영:

| 구분 | Q&A 튜토리얼 | 위키 |
|------|-------------|------|
| 형식 | Q{N}: 질문 → 내러티브 답변 | 개념·엔티티 페이지 + [[link]] |
| 트리거 | 사용자 질문 | 정리된 지식 베이스 |
| 갱신 빈도 | 매 질문 시 누적 | 주요 개념·결과 변경 시 |
| 권위 | 시간순 학습 기록 | 컴파일된 best understanding |

질문이 들어오면 먼저 위키에 답이 있는지 확인, 그 후 답변 후 Q&A에 추가, 필요시 위키도 갱신하는 흐름.

### 답변 작성 시

- **개념 설명**: 위키 페이지 우선 인용, 없으면 외부 보강 후 raw 드롭 권장
- **검증 대상 모델 사양**: `wiki/<model>.md` 사용 (config 수치 모두 포함됨)
- **방법론 비교**: `wiki/tokenizer-analysis.md` 등 4종 + `model-provenance-testing.md`
- **사업 맥락**: `wiki/dokpamo-project.md` + `from-scratch-debate.md`
- **인프라 작업**: `wiki/sagemaker-spot-training.md`, `aws-cdk-typescript.md`, `huggingface-hub-usage.md`

### 정직성

위키 confidence가 `low` 또는 `tentative`인 페이지를 인용할 때는 답변에 명시. lint로 정기 점검(`python3 ~/.claude/plugins/cache/roboco-plugins/documentation/0.3.0/skills/llm-wiki/scripts/lint_wiki.py --wiki-dir wiki`).

## 안티패턴

- 위키 인덱스 무시하고 외부 검색부터 — 시간·비용 낭비
- 단일 페이지만 읽고 답변 — `[[link]]`로 맥락 확장 필요
- `wiki/index.md` 수동 편집 — 다음 sync에 덮어씌워짐
- `wiki/raw/` 파일 수정 — 불변 원칙. 새 정보는 새 파일로 드롭
- 위키 업데이트 없이 새로운 결론을 답변에만 — 다음 세션이 같은 외부 검색을 또 해야 함
