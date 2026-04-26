---
title: "독파모 사업 (국가 AI 파운데이션 모델)"
type: entity
tags: [korean-government, msit, dokpamo, policy]
status: active
confidence: high
created: 2026-04-26
updated: 2026-04-26
sources: [raw/2026-04-26_dokpamo-status.md]
---

# 독파모 사업 (국가 AI 파운데이션 모델)

대한민국 과학기술정보통신부(MSIT) 주관, 2025-08 출범. 한국이 미·중에 이은 AI 강국 진입을 목표로 글로벌 95% 성능 동등 달성을 위해 5개 컨소시엄에 6개월 단위 단계 평가를 통해 자원을 집중하는 사업.

## 평가 단계

5 → 4 → 3 → 2, 약 6개월마다. 2027년 말 최종 1~2개 모델 선정.

### 1차 평가 (2026-01-09)

| 순위 | 컨소시엄 | 모델 | 결과 |
|------|----------|------|------|
| 1 | LG AI 연구원 | [[k-exaone-236b]] | Phase 2 진출 |
| 2 | SKT | [[ax-k1]] | Phase 2 진출 |
| 3 | Upstage | [[solar-open-100b]] | Phase 2 진출 |
| 탈락 | NAVER Cloud | HyperCLOVAX | Phase 1 탈락 |
| 탈락 | NC AI | VAETKI/배키 | Phase 1 탈락 |

### 보충 선발 (2026-02)

탈락팀 발생으로 경쟁 다양성 확보 차원. Motif Technology(약 300B)와 Trillion Labs 경합 → **Motif Technology 추가 선정**. 현재 Phase 2는 4팀 (LG, SKT, Upstage, Motif).

## 예산 (2026)

| 항목 | 규모 |
|------|------|
| 정부 R&D 총예산 中 AI | **9.9조원** (전년 대비 3배) |
| 단기 GPU 임대 | **1,576억원** |
| 정부 직접 구매 GPU | **최대 10,000장** (2026 중반부터 분배) |
| 팀별 데이터 생성·전처리 | **30~50억원/년** |
| 해외·재외 한인 인재 매칭 | **약 20억원/년/팀** (2027까지 보장) |

## 외부 검증

**Stanford AI Index 2026 (4월 23일 정정판)**: 한국 모델 8개 등재로 글로벌 4위→3위. 1위 미국 50개, 2위 중국 30개. 8개 중 5개가 독파모 산출물 ([[solar-open-100b]], EXAONE 4종, [[ax-k1]], HyperCLOVAX-SEED-32B-Sync, VAETKI/배키).

## 정책 연계

- **K-Moonshot (2026-02-25 발표)**: 바이오·소재·반도체 등 도메인 특화 파운데이션 모델 별도 개발 (2027~2031, 4,640억원). 독파모를 인프라로 두고 그 위에 도메인 모델을 쌓는 계층 전략.
- **지역 AX**: 4대 권역에 2026~2030년 3.1조원 투입.

## 미해결 이슈

- **From scratch 정량 기준 미정**: [[from-scratch-debate]] 참조. 차기 라운드의 잠재 리스크.
- **상용성 vs 오픈소스**: 정부 지원 종료 후 사업화 경로 불투명.
- **NAVER/NC 분리 트랙**: 탈락팀이 자체 모델 개발 계속 (HyperCLOVAX Dash 등).

## 본 프로젝트 위치

본 프로젝트는 **독파모 사업의 official verifier가 아닌 외부 비전문가의 자기주도 검증**. 5개 컨소시엄 모델의 from scratch 주장에 대해 [[tokenizer-analysis]], [[architecture-analysis]] 등 publicly available 방법으로 검증 시도.

본 프로젝트의 LLM-DNA 단계는 **차기 라운드 진출 3종**([[solar-open-100b]], [[k-exaone-236b]], [[ax-k1]])에 집중하여 functional 계통을 추가 분석.

## 관련 페이지

- [[from-scratch-debate]] — 사업의 핵심 정의 이슈
- [[solar-open-100b]], [[k-exaone-236b]], [[ax-k1]] — Phase 2 진출 모델
- [[llm-dna-overview]] — 본 프로젝트의 새 검증 도구
