# 2차평가 진출 모델 공개 자료 조사

조사일: 2026-10-12. 공개본과 실제 평가 제출 체크포인트의 동일성은 확인하지 못했다.

- [연합뉴스, 2026-08-18](https://www.yna.co.kr/view/AKR20260818120600017): 진출 3개사와 Solar Open 2, A.X K2, K-EXAONE 2.0 명칭 확인.
- [Solar Open 2 보고서 v2 §3.1](https://arxiv.org/html/2607.20062v2): 이전 Solar Open 1 가중치 5.69B를 선택 이전한다고 설명.
- [K-EXAONE 2.0 보고서 §2.2](https://arxiv.org/html/2608.04505): 이전 K-EXAONE을 확장한 뒤 계속 사전학습한다고 설명.
- [A.X K2 보고서](https://arxiv.org/html/2608.30181): From Scratch 학습 주장. 독립 검증된 초기화 기록과 구분해야 한다.
- 원 개발사 공개본: [Solar 2](https://huggingface.co/upstage/Solar-Open2-250B), [A.X K2](https://huggingface.co/skt/A.X-K2), [EXAONE 2](https://huggingface.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B).

측정 결과와 리비전은 [증거 JSON](2026-10-12_second-round-evidence.json), 해석과 한계는 [분석 보고서](../../docs/07-second-round-analysis.md)에 보존했다. 모든 merge 규칙의 계승은 세 쌍에서 확인했지만 가중치 재사용의 충분조건은 아니다. 가중치 표본은 Solar·EXAONE의 공개된 계승 경로와 부합했다. A.X K2의 선택 BF16 embedding/head 표본은 K1과 낮은 상관을 보였다. 이 결과는 전체 초기화의 독립 확증이 아니다.
