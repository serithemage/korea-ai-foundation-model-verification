# 변경 이력 (Changelog)

이 문서는 프로젝트의 주요 변경 사항을 시간순으로 기록합니다.

> 전체 커밋 이력은 [GitHub Commits](https://github.com/serithemage/korea-ai-foundation-model-verification/commits/main)에서 확인할 수 있습니다.

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
