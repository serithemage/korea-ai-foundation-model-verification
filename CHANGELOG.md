# 변경 이력 (Changelog)

이 문서는 프로젝트의 주요 변경 사항을 시간순으로 기록합니다.

> 전체 커밋 이력은 [GitHub Commits](https://github.com/serithemage/korea-ai-foundation-model-verification/commits/main)에서 확인할 수 있습니다.

---

## 2026-10-12

### 모델 검증

- 2차평가 진출 Solar Open 2·A.X K2·K-EXAONE 2.0의 공개 자료, 전체 토크나이저, 선택 BF16 가중치 표본을 비교하고 독립 확증의 한계를 기록.
- AWM·GhostSpec 기반 CPU 시범 검사로 embedding 정렬·미사용 행 확인·선택 GQA 층 UCKA/스펙트럼을 계산하고 원본 수치·재현 스크립트·수학 검증 3개를 보존; AWS 사용 비용 $0.

### 문서 추가/수정

- [2차평가 예비검증 보고서](docs/07-second-round-analysis.md), 고정 리비전·파일 해시·가중치 원본 표본 증거, 출처 노트를 추가하고 README의 세대별 판정을 분리.
- Exa로 추가 판별 방법의 핵심 원문 8개를 조사하고 [방법론 보고서](docs/08-from-scratch-methods.md)와 출처 노트를 추가.
- 모바일 대응 정적 보고서 페이지를 추가하고 GitHub Pages 공개 경로를 README에 연결.
- 지정 Spot 대시보드와 AWS 읽기 API로 리전·AZ·가격·quota를 확인하고 승인 전 확대 검사 견적을 작성.
- 공개 페이지의 세 모델에 방법별 근거·결과 해설·상세 문서/증거 링크를 추가하고 시스템·라이트·다크 테마 전환과 선택 저장을 구현.

### 추가 검사 진행

- 사용자 승인 $15 한도에서 오하이오 EC2 Spot으로 GPU 확대 검사를 시작; Spot 전용 요청과 110분 자동 종료·외부 종료 감시를 설정하고 결과 판정은 완료 후 기록.

### 방법론 개선

- 전체 무작위 초기화와 자체 선행 모델 계승을 구분하고, 구조 불일치·표본 해시 불일치를 재사용 배제 증거로 사용하지 않음.

---

## 2026-01-05

### HyperCLOVAX 정부 가이드라인 맥락 추가
- Vision Encoder 재사용이 정부 가이드라인 위반인지 조사
- MSIT 공식 발표 참조 추가
- VLM 업계 관행 (CLIP/SigLIP 재사용) 설명 추가
- [config.json](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B/blob/main/config.json) 직접 링크 추가

### 검증 방법론 상세 설명 추가
- Tokenizer, Weight, Architecture, 행동, Training Logs 분석 방법 상세화
- 각 방법론별 판정 기준표 추가

### 5개 모델 검증 완료
- **LG AI K-EXAONE**: From scratch 신뢰 가능
- **NC AI VAETKI**: From scratch 신뢰 가능
- **SKT A.X-K1**: From scratch 신뢰 가능
- **HyperCLOVAX-SEED**: Vision Encoder 재사용 확인, Text Decoder 추가 검증 필요
- **Solar-Open-100B**: From scratch 신뢰 가능 (이전 완료)

### HyperCLOVAX vocab_size 크로스 체크
- Llama 3 (128,000)와 256 토큰 차이 확인
- Trillion-7B 논문 참조로 한국어 최적화 설계 가능성 확인
- Q10 추가: vocab_size 일치의 중요성 설명

### 방법론 한계 및 학술 연구 반영
- 검증 방법론 비판 조사: vocab_size 비교 vs 실제 토큰 중복률/weight 비교
- [arXiv:2502.00706v1](https://arxiv.org/abs/2502.00706) Black-box Output Similarity Testing 방법론 참조 추가
- Yi-Llama 논쟁 사례 연구 추가 (아키텍처 유사성 ≠ 파생 증거)
- README.md에 "방법론의 한계와 학술 연구 결과" 섹션 추가
- 튜토리얼 Q11: 방법론 비판에 대한 학술적 근거 추가

### 프로젝트 자동화
- `.claude/skills/update-changelog.md` skill 생성
- 변경이력 자동 업데이트 기능 추가

### HyperCLOVAX-SEED Tokenizer 심층 분석 (이슈 #5)
- **실제 vocab_size 정정**: config.json (128,256) vs tokenizer.json (110,524)
- HyperCLOVAX-SEED vs Llama-3 비교:
  - 토큰 중복률: 92.58%
  - **BPE merge rules 순서 일치율: 0.01%** (완전 불일치)
  - 결론: Llama 3 직접 재사용 ❌
- HyperCLOVAX-SEED vs Qwen2.5-VL 비교:
  - 토큰 중복률: 91.23%
  - **BPE merge rules 순서 일치율: 1.54%** (Llama보다 150배 높음)
  - **처음 20개 merge rules 완전 일치**
  - 결론: Qwen 계열과 유사한 BPE 학습 패턴 ⚠️
- `docs/01-tokenizer-analysis.md` HyperCLOVAX 섹션 업데이트

### 네이버 오픈소스 차용 공식 인정 반영
- [뉴스1 단독 보도](https://www.news1.kr/it-science/general-it/6029076) 내용 반영
- 네이버 인정 내용:
  - **Vision Encoder**: Qwen 2.5 파인튜닝 사용 (코사인 유사도 99.51%)
  - **Audio Encoder**: 원본 그대로 사용
  - **Text Decoder**: "100% 자체 기술" 주장
  - **Tokenizer**: 공식 언급 없음
- 전문가 의견 (장두성 서강대 교수): "업계 관행이나 '프롬 스크래치'라고 하지는 않는다"
- 네이버 인정 내용이 Tokenizer 분석 결과의 해명은 아님 (별도 해명 필요)

---

## 2026-01-04

### Solar-Open-100B 검증 완료
- Tokenizer 분석: vocab_size 196,608 (모든 모델과 불일치)
- Architecture 분석: 48 layers, 129 experts (고유 구성)
- Weight 분석: Architecture 불일치로 비교 불가 (간접 지지)
- 행동 분석: 공개 검증 세션 + LayerNorm 의혹 독립 검증으로 해소

### LayerNorm 유사도 의혹 해소
- [hyunwoongko 독립 검증](https://github.com/hyunwoongko/solar-vs-glm-vs-phi) 결과 반영
- 96.8% 유사도 주장이 방법론적 오류(초기화 편향)였음 확인

### 프로젝트 구조 확립
- 5개 모델 통합 검증 구조로 문서 재구성
- 튜토리얼 및 검증 방법론 문서 분리
- Q&A 형식 학습 기록 시스템 구축

---

## 프로젝트 시작

### 초기 설정
- Solar-Open-100B 검증 프로젝트 시작
- 리포지토리명: `solar-open-100b-scratch-verification` → `korea-ai-foundation-model-verification`으로 변경
- 검증 방법론 5가지 정의: Tokenizer, Weight, Architecture, 행동, Training Logs
