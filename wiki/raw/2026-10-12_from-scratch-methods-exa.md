# Exa 조사: 추가 From Scratch 판별 방법

조사일: 2026-10-12. 네 검색 관점에서 각 5개, 총 20개 결과를 검토하고 핵심 원문 8개를 추가로 읽었다. 검색 결과 수는 독립적으로 검증된 논문 수와 다르다. 검색 결과는 후보 발견에 사용하고 핵심 주장은 저자 논문에서 확인했다.

검색 관점:

- 공개 가중치에서의 provenance 및 permutation 불변 지문.
- 무작위 초기화부터의 Proof-of-Learning과 학습 transcript 검증.
- black-box 출력 및 특화 prompt 기반 계통 지문.
- fine-tuned 모델의 부모 식별과 representation alignment.

핵심 출처:

- [AWM](https://arxiv.org/abs/2510.06738): embedding 정렬과 unbiased CKA 기반 가중치 비교.
- [GhostSpec](https://arxiv.org/abs/2511.06390): attention 행렬 곱의 스펙트럼과 layer 정렬.
- [REEF](https://arxiv.org/abs/2410.14273): 동일 입력의 representation CKA.
- [HuRef](https://papers.nips.cc/paper_files/paper/2024/file/e46fc33e80e9fa2febcdb058fba4beca-Paper-Conference.pdf): 변환 불변 항으로 구성한 가중치 지문과 검증 가능한 지문 생성.
- [Model Provenance Testing](https://proceedings.neurips.cc/paper_files/paper/2025/file/30e55dfbfa8e76e3526de7f2ab029cda-Paper-Conference.pdf): unrelated baseline을 둔 출력 가설검정.
- [LLMPrint](https://aclanthology.org/2026.acl-long.541/): 최적화 prompt로 token 선호 지문 생성.
- [Proof-of-Learning](https://arxiv.org/abs/2103.05633): 학습 업데이트·checkpoint 경로 검증.
- [Proof-of-Training-Data](https://proceedings.neurips.cc/paper_files/paper/2023/file/03e33e1f62e3302b47fe1d38a235921e-Paper-Conference.pdf): seed commitment 및 transcript 검사와 PoL 공격 대응.

모두 원 저자의 연구이며 논문에서 보고한 성능을 이번 250B~750B 모델에 외삽하지 않는다. 상세 해석과 적용 순서는 [조사 보고서](../../docs/08-from-scratch-methods.md)에 기록했다. 이번에는 방법을 조사했으며 새 방법을 대상 모델에 실행하지 않았다.
