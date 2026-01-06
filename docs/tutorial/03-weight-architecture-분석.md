# Weight 및 Architecture 분석

Weight 비교와 architecture 분석 방법을 다룹니다.

---

## Q3: Weight 분석이란 무엇이고, 어떻게 from scratch 여부를 판별하나요?

**질문 시각**: 2026-01-04

**답변**:

Weight 분석은 모델의 가중치를 직접 비교하여 from scratch 학습 여부를 판별하는 방법입니다. Fine-tuned 모델은 base model과 높은 가중치 유사성을 보이는 반면, from scratch 모델은 독립적인 가중치 분포를 갖습니다.

첫 번째 방법은 Layer별 Cosine Similarity 분석입니다. 두 모델의 대응하는 weight matrix 간 방향적 유사성을 측정합니다. Fine-tuned 모델은 초기 레이어에서 0.95 이상의 유사도를 보이며 후기 레이어로 갈수록 점차 감소합니다. 반면 from scratch 모델은 전체적으로 0에 가까운 낮은 유사도를 보입니다.

두 번째 방법은 Weight Tensor 해싱입니다. 대규모 모델에서 효율적인 비교를 위해 해시 기반 fingerprint를 사용합니다. MinHash나 SimHash로 Jaccard 유사도를 계산하거나, Tensor Checksum으로 양자화된 weight의 perceptual hash를 비교합니다. MSE가 1e-5 이내면 동일 layer로 판정합니다. 동일한 해시가 나오면 동일 weight이므로 fine-tuning의 강력한 증거입니다.

세 번째 방법은 PCA 분석입니다. 고차원 weight matrix를 저차원으로 투영하여 분포를 비교합니다. Fine-tuned 모델은 base model 근처에 clustering되어 80% 이상 overlap됩니다. From scratch 모델은 완전히 다른 cluster를 형성합니다.

네 번째 방법은 Embedding Layer 분석입니다. Token embedding은 fine-tuning에서 가장 적게 변하므로 특히 중요합니다. Embedding matrix의 cosine similarity를 비교하거나, K-means clustering으로 centroid를 비교합니다. 흥미로운 점은 shape가 불일치하면, 즉 다른 vocabulary를 사용한다면 그 자체로 from scratch 증거가 됩니다.

유사도 임계값을 보면, fine-tuning의 경우 평균 Layer Cosine Sim이 0.90 이상이고 초기 layer는 0.99 이상입니다. Embedding Cosine Sim은 0.98 이상입니다. From scratch의 경우 평균 Layer Cosine Sim이 0.3 미만이고 Embedding Cosine Sim은 0.1 미만입니다.

100B MoE 모델을 분석할 때는 몇 가지 실제적인 도전 과제가 있습니다. 메모리와 연산량 측면에서 100B MoE 모델은 full weight 로드에 1TB 이상의 RAM이 필요한데, sharded loading이나 FP8 양자화로 해결할 수 있습니다. MoE 구조의 특수성 때문에 router weight와 expert weight를 별도로 분석해야 합니다. Architecture가 불일치하면 직접 비교가 불가능하므로 num_experts, head_dim 등을 먼저 확인해야 합니다. 대규모 tensor에서는 precision error가 누적될 수 있어서 double precision을 사용하거나 10-20% submatrix를 샘플링합니다.

---

## Q4: Cosine Similarity 분석이란 무엇이고, LLM weight 비교에 어떻게 활용되나요?

**질문 시각**: 2026-01-04

**답변**:

Cosine Similarity는 두 벡터 간의 방향적 유사성을 측정하는 지표로, LLM weight 비교에서 핵심적인 역할을 합니다. 수학적으로는 두 벡터의 내적을 각 벡터의 L2 norm으로 나눈 값입니다. 결과는 -1에서 1 사이이며, 양수 weight의 경우 0에서 1 사이입니다.

Weight 비교에 Cosine Similarity를 사용하는 이유는 여러 가지입니다. 우선 scale 불변성이 있어서 벡터 크기에 독립적으로 방향만 비교합니다. 고차원에서도 효율적으로 계산이 가능하고, 1에 가까울수록 유사하고 0에 가까울수록 다르다는 해석이 직관적입니다. 별도의 normalize 과정 없이 자체적으로 정규화가 포함됩니다.

값의 해석을 보면, 0.99에서 1.0 사이는 거의 동일한 weight로 fine-tuning의 강력한 증거입니다. 0.90에서 0.99는 높은 유사도로 fine-tuning이거나 같은 initialization일 가능성을 시사합니다. 0.50에서 0.90은 중간 유사도로 부분적 공유 가능성이 있습니다. 0.10에서 0.50은 낮은 유사도로 독립적 학습 가능성이 있습니다. 0.0에서 0.10은 거의 무관한, orthogonal한 관계로 from scratch의 강력한 증거입니다.

Fine-tuned 모델의 전형적인 layer별 패턴을 살펴보면 흥미롭습니다. Embedding인 Layer 0은 0.98에서 0.99로 거의 변화가 없습니다. 초기 layer인 Layer 1-5는 0.95에서 0.99로 약간의 조정만 있습니다. 중간 layer인 Layer 6-20은 0.85에서 0.95로 점진적으로 감소합니다. 후기 layer인 Layer 21 이상은 0.70에서 0.90으로 task-specific 학습이 반영됩니다. Output Layer는 0.60에서 0.85로 가장 많이 변화합니다. 반면 from scratch 모델은 모든 layer에서 0.0에서 0.3 수준의 낮은 유사도를 보입니다. 이는 random initialization의 효과입니다.

대규모 모델에서는 실용적인 고려사항이 있습니다. 메모리 최적화를 위해 전체 tensor를 한 번에 로드하지 않고 chunk 단위로 처리하며, FP16이나 BF16으로 계산하여 메모리를 절약합니다. 샘플링 전략도 필요한데, 100B 이상 모델은 전체 weight 비교가 비실용적이므로 layer당 10-20%를 무작위 샘플링하여 추정하고 통계적 신뢰구간을 계산합니다. MoE 모델은 router weight와 expert weight를 분리 분석하고, shared expert와 routed expert를 구분해야 합니다.

Cosine Similarity에도 한계가 있습니다. Zero vector에서는 정의되지 않으므로 zero 체크 후 처리해야 합니다. 방향만 비교하고 magnitude는 무시하므로 L2 distance를 병행하면 좋습니다. 고차원에서 값이 중앙으로 수렴하는 경향이 있어서 layer별 분석으로 보완합니다. 극단값에 영향을 받을 수 있어서 robust 버전을 사용하기도 합니다.

---

## Q7: Solar-Open-100B의 Weight 분석이 가능한가요? Architecture 비교 결과는?

**질문 시각**: 2026-01-04

**답변**:

Weight 분석을 수행하기 전에 먼저 architecture 비교를 통해 weight 비교가 가능한지 확인해야 합니다. Weight 비교는 동일한 shape의 tensor 간에만 의미가 있기 때문입니다.

Solar-Open-100B의 architecture를 다른 MoE 모델들과 비교해보면 흥미로운 결과가 나옵니다. hidden_size는 Solar가 4,096으로 Mixtral과 동일하지만, DeepSeek-V2는 5,120이고 Qwen2-57B는 3,584입니다. num_hidden_layers는 Solar가 48개인데, Mixtral은 32개, DeepSeek은 60개, Qwen2는 28개로 모두 다릅니다. num_attention_heads는 Solar가 64개로 Mixtral의 32개, DeepSeek의 128개, Qwen2의 28개와 모두 다릅니다.

가장 중요한 MoE 관련 파라미터를 보면, n_routed_experts는 Solar가 128개, Mixtral이 8개, DeepSeek이 160개, Qwen2가 64개입니다. n_shared_experts는 Solar가 1개, Mixtral은 0개, DeepSeek은 2개입니다. num_experts_per_tok는 Solar가 8개, Mixtral이 2개, DeepSeek이 6개, Qwen2가 8개입니다.

이런 차이 때문에 어떤 모델과도 weight 비교가 불가능합니다. Mixtral과는 hidden_size가 동일하지만 layer 수와 expert 수가 다릅니다. DeepSeek-V2, Qwen2-57B와는 hidden_size부터 다릅니다.

구체적으로 보면, Embedding Layer는 Solar가 [196,608, 4,096] shape인데 Mixtral은 [32,000, 4,096], DeepSeek은 [102,400, 5,120], Qwen2는 [151,936, 3,584]입니다. vocab_size와 hidden_size가 모두 다르므로 embedding weight 비교가 불가능합니다. Attention Layer도 마찬가지로 Q, K, V projection matrix shape가 모두 다릅니다. MoE Layer 역시 expert 수와 intermediate_size가 모두 다릅니다.

Weight 비교가 불가능하다는 사실 자체가 from scratch의 증거입니다. Fine-tuning된 모델이라면 base model과 동일한 architecture를 가져야 합니다. Solar-Open-100B는 어떤 기존 MoE 모델과도 architecture가 일치하지 않습니다. 48 layers, 128+1 experts, 196k vocab이라는 고유한 구성을 가지고 있습니다.

특히 129개 expert 구성이 독특합니다. 128개의 routed expert와 1개의 shared expert를 사용하는 이 구성은 다른 모델에서 볼 수 없습니다. moe_intermediate_size도 1,280으로 가장 작습니다. Layer 수 48도 Mixtral의 32와 DeepSeek의 60 사이이지만, 정확히 중간값이 아닙니다.

결론적으로 architecture가 완전히 다르기 때문에 weight 비교는 불필요하며, 이 자체가 from scratch 학습의 강력한 증거입니다.

---

## Q12: 왜 많은 대규모 MoE 모델들이 48 레이어와 129 experts를 사용하나요?

**질문 시각**: 2026-01-06

**답변**:

DeepSeek이나 Solar 같은 대규모 MoE 모델들이 유사한 구성을 사용하는 이유가 궁금할 수 있습니다. 결론부터 말하면, 48 레이어와 129 experts는 수학적 공식에서 도출된 것이 아니라 경험적 최적화의 결과입니다.

먼저 48 레이어를 사용하는 이유를 살펴보겠습니다. MoE 모델에서 FFN(Feed-Forward Network) 레이어가 전체 파라미터의 대부분을 차지합니다. 레이어 수는 "총 파라미터 대비 활성 파라미터" 비율을 조절하는 핵심 변수입니다. 예를 들어 DBRX는 132B 총 파라미터 중 36B만 활성화됩니다. 48이라는 숫자는 2⁴ × 3으로, GPU/TPU 클러스터에서 레이어를 효율적으로 분배할 수 있는 구조입니다. 다양한 분할이 가능해서 하드웨어 병렬화에 유리합니다.

129 experts를 사용하는 이유는 더 흥미롭습니다. 129는 128+1 구조로 이해할 수 있습니다. 128은 2⁷로 power of 2이기 때문에 메모리와 연산 최적화에 유리합니다. 그리고 +1은 Shared Expert입니다. 모든 토큰이 공유하는 전문가로, DeepSeek-V2와 V3에서 이 패턴을 사용합니다.

Fine-grained MoE의 이점도 있습니다. 더 많은 작은 experts는 더 많은 조합 가능성을 의미합니다. 129개 중 top-k(보통 6-8개)를 활성화하면, 8개 experts만 사용하는 모델보다 65배 이상 많은 조합이 가능합니다. 각 expert가 특정 도메인이나 패턴에 집중하는 전문화가 극대화되고, 중복이 감소하면서 품질이 향상됩니다.

실제 사례를 비교해보면, DeepSeek-V3는 257개(256+1) experts 중 8개를 활성화하고 61개 레이어를 사용합니다. Qwen2.5-MoE는 64개 중 8개를 활성화합니다. Mixtral은 8개 중 2개를 활성화하고 32개 레이어를 사용합니다. DBRX는 16개 중 4개를 활성화합니다. 모델마다 다른 구성을 사용하지만, 최근 대규모 모델들은 더 많은 experts와 shared expert 패턴으로 수렴하는 경향이 있습니다.

결론적으로 48 레이어와 129 experts 조합은 스케일링 법칙, 하드웨어 효율성, 품질-비용 트레이드오프를 경험적으로 최적화한 결과입니다. 특히 129 = 128+1 구조는 Shared Expert 패턴의 전형적인 표현입니다.

---

<!-- SECTION_MARKER: 새로운 Weight/Architecture 분석 Q&A는 이 마커 위에 추가됩니다 -->
