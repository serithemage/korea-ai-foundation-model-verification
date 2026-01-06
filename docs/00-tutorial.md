# 튜토리얼: Q&A 형식 학습 기록

이 문서는 클로드 스킬을 이용해서 국가 AI 파운데이션 모델 검증 프로젝트(구 Solar-Open-100B 검증 프로젝트)를 진행하면서 던진 질문과 답변을 시간순으로 기록한 것입니다.

---

## Q0: 왜 "from scratch" vs "fine-tuning" 논란이 발생하나요?

**질문 시각**: 2026-01-05

**답변**:

한국 정부의 국가 AI 파운데이션 모델 프로젝트는 "from scratch 학습"을 필수 요건으로 규정하고 있습니다. 여기서 from scratch란 무작위로 초기화된 weight에서 시작하여 전체 학습을 수행하는 것을 의미하며, 기존 모델의 pre-trained weight를 가져와 추가 학습하는 fine-tuning과는 근본적으로 다릅니다.

이 구분이 중요한 첫 번째 이유는 AI 주권 문제입니다. 기존 모델의 weight를 사용하면 해당 모델의 라이선스와 제약에 종속되기 때문에, 국가 전략 기술로서 독자적인 기술력을 보유하려면 from scratch 학습이 필수적입니다. 향후 모델을 수정하거나 확장할 때의 자유도 역시 from scratch일 때만 확보됩니다.

두 번째 이유는 국민 세금 사용과 관련됩니다. 국가 프로젝트는 정부 예산으로 진행되는데, fine-tuning은 from scratch 대비 훨씬 적은 비용으로 가능합니다. 학습 토큰 수로 비교하면 from scratch는 수조에서 수십조 토큰이 필요하지만, fine-tuning은 수십억에서 수백억 토큰이면 충분합니다. GPU 시간도 from scratch는 수만에서 수십만 GPU-hours가 드는 반면, fine-tuning은 수백에서 수천 GPU-hours로 가능합니다. 비용으로 환산하면 from scratch는 수백억에서 수천억 원이 소요되지만, fine-tuning은 수억에서 수십억 원 수준입니다. 따라서 from scratch 비용을 받고 fine-tuning만 했다면 이는 예산 낭비 또는 사기 논란으로 이어질 수 있습니다.

세 번째 이유는 신뢰도와 평판입니다. "from scratch"라고 주장했는데 실제로는 fine-tuning이었다면 심각한 신뢰 손상이 발생합니다. 이는 국제 AI 커뮤니티에서 한국 AI 기술력의 평판에 영향을 미치고, 후속 프로젝트 및 투자 유치에도 부정적 영향을 줍니다.

---

## Q1: LLM이 "from scratch"로 학습되었는지 어떻게 검증할 수 있나요?

**질문 시각**: 2026-01-04

**답변**:

LLM이 from scratch로 학습되었는지 검증하는 방법은 여러 가지가 있으며, 각각 장단점이 있습니다.

가장 접근성이 높은 방법은 Tokenizer 분석입니다. Tokenizer는 재학습 비용이 높아서 fine-tuning 시에는 거의 재사용됩니다. 따라서 vocabulary의 95% 이상이 기존 모델과 동일하면 fine-tuning 가능성이 높습니다. `tokenizer.get_vocab()`으로 vocabulary를 추출하여 비교할 수 있습니다.

Weight 분석도 효과적인 방법입니다. Fine-tuned 모델은 초기 레이어에서 base model과 90% 이상의 cosine similarity를 유지하는 경향이 있습니다. SHA-256 해시로 weight tensor를 비교하거나, PCA 분석으로 weight 분포를 시각화할 수 있습니다. From scratch 모델은 PCA에서 orthogonal한 분포를 보입니다.

Architecture 비교는 `model.config`로 hyperparameter를 확인하는 방식입니다. 동일한 config는 fine-tuning의 강력한 증거이며, 고유한 구성요소가 있다면 from scratch 증거가 됩니다. 예를 들어 특이한 RoPE scaling 설정 같은 것이 그렇습니다.

행동 테스트는 Knowledge cutoff 날짜를 확인하거나, Safety alignment와 refusal pattern이 base model과 동일한지 살펴보는 방식입니다. 이런 패턴이 유사하면 fine-tuning 가능성이 있습니다.

Compute 추정도 간접적인 증거가 됩니다. From scratch는 fine-tuning 대비 10-100배 더 많은 compute가 필요합니다. 예를 들어 19.7T tokens 학습은 massive compute를 요구하므로, from scratch 주장과 일관성이 있습니다.

신뢰도 측면에서 보면, Training Logs가 가장 높은 신뢰도를 가지지만 접근성이 낮습니다. Tokenizer 분석은 신뢰도와 접근성이 모두 높아서 첫 번째 검증 수단으로 적합합니다.

---

## Q2: Tokenizer 분석이란 무엇이고, 어떻게 from scratch 여부를 판별하나요?

**질문 시각**: 2026-01-04

**답변**:

Tokenizer는 텍스트를 모델이 처리할 수 있는 숫자(token ID)로 변환하는 역할을 합니다. 주요 방식으로는 BPE(Byte Pair Encoding), WordPiece, SentencePiece가 있습니다. BPE는 빈도 기반으로 인접 문자쌍을 병합하며 GPT-2나 RoBERTa에서 사용됩니다. WordPiece는 likelihood 최대화를 기준으로 병합하며 BERT에서 사용됩니다. SentencePiece는 공백을 포함한 원시 텍스트를 처리하며 ▁ 마커를 사용하는데, T5, Gemma, Llama 등에서 채택하고 있습니다.

Fine-tuning 시 tokenizer를 재학습하지 않는 이유가 있습니다. 우선 embedding 호환성 문제가 있어서, 새 vocabulary는 pre-trained embedding과 호환되지 않습니다. 또한 tokenizer 재학습은 전체 corpus를 재처리해야 하므로 비용이 많이 듭니다. 게다가 vocabulary를 변경하면 원래 도메인에서 성능이 저하될 위험도 있습니다.

Vocabulary 중복률로 판단할 때, 98% 이상이면 fine-tuning 가능성이 높습니다. 이는 safety token 등 소량만 추가한 경우에 해당합니다. 90-98%는 continued pre-training이나 vocabulary 확장을 시사합니다. 90% 미만이면 from scratch 학습의 강력한 증거가 됩니다.

분석 기법으로는 먼저 vocabulary 비교가 있습니다. 두 tokenizer의 vocab을 추출하여 교집합을 구하고 중복률을 계산합니다. 그 다음 merge rules 비교가 있는데, BPE나 SentencePiece에서 merge 순서까지 같으면 동일한 tokenizer입니다. Special tokens 비교도 중요한데, `[PAD]`, `[UNK]`, `[CLS]`, `<eos>` 등의 패턴을 살펴봅니다. 마지막으로 encoding 결과 비교가 있어서, 동일한 입력에 대해 다른 토큰 분할이 나오면 다른 tokenizer임을 알 수 있습니다.

Solar-Open-100B를 검증할 때는 Llama 계열, Mistral/Mixtral, Qwen, DeepSeek-MoE 등과 비교합니다. 이 모델들과 90% 미만의 중복률이 나오면 from scratch 주장이 지지됩니다.

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

## Q5: Solar-Open-100B의 Tokenizer 분석 결과는 어떠한가요?

**질문 시각**: 2026-01-04

**답변**:

Solar-Open-100B의 tokenizer를 주요 base model들과 비교 분석했습니다. 결론부터 말하면, from scratch 학습 주장을 강력히 지지하는 결과입니다.

Solar-Open-100B의 vocabulary 크기는 196,608개입니다. 이를 다른 모델들과 비교해보면 흥미로운 패턴이 드러납니다. 가장 가까운 Qwen2-72B가 152,064개로 Solar보다 29% 작습니다. Llama-3는 128,256개로 Solar보다 53% 작고, DeepSeek-V2는 102,400개로 92% 작습니다. Mixtral-8x7B는 32,000개에 불과해서 Solar보다 514%나 작습니다.

이렇게 큰 차이가 나는 것은 vocabulary를 재사용하지 않고 독립적으로 학습했음을 강력히 시사합니다. 만약 기존 모델을 fine-tuning했다면, vocabulary 크기가 같거나 약간의 special token만 추가된 수준이어야 합니다.

Special token 패턴도 살펴보았습니다. Solar-Open-100B는 `<s>`, `</s>`, `<pad>` 형식을 사용합니다. 이는 SentencePiece의 전통적인 방식입니다. Llama-3는 `<|begin_of_text|>`, `<|end_of_text|>` 같은 새로운 형식을 도입했고, Mixtral은 Solar와 비슷하게 `<s>`, `</s>`를 사용하지만 vocab_size가 6배 이상 차이납니다.

Tokenizer type을 보면, Solar-Open-100B와 Mixtral 모두 SentencePiece BPE를 사용합니다. 하지만 같은 방식을 사용한다고 해서 같은 tokenizer는 아닙니다. Llama-3는 tiktoken BPE라는 다른 구현을 사용합니다.

종합하면, vocabulary 크기가 어떤 주요 base model과도 일치하지 않는다는 점이 가장 결정적입니다. 이 규모의 차이는 vocabulary 확장이나 fine-tuning으로는 설명되지 않습니다. 독립적으로 학습된 tokenizer임이 강력히 시사됩니다.

더 확실한 결론을 위해서는 실제 vocabulary 토큰 목록을 다운로드하여 중복률을 계산하고, BPE merge rules 순서를 비교하며, 동일 텍스트에 대한 토큰화 결과를 비교하는 추가 검증이 필요합니다.

---

## Q6: Fine-tuning 의심 증거들에 대한 반론은 무엇인가요?

**질문 시각**: 2026-01-04

**답변**:

Fine-tuning 의심 증거로 제시되는 기준들에 대해, 실제로는 from scratch인데 fine-tuning으로 오판될 수 있는 경우를 분석해보겠습니다.

첫 번째로 "95% 이상 vocabulary 중복 = Fine-tuning"이라는 주장에 대한 반론입니다. 동일한 언어 분포로 학습하면 자연스럽게 유사한 빈도 패턴이 발생합니다. 영어 웹 데이터로 학습하면 독립적으로 학습해도 vocabulary가 비슷해질 수 있습니다. 또한 vocab_size=32k, character_coverage=0.9995 같은 표준 설정을 사용하면 유사한 결과가 나올 수 있습니다. 자연어의 공통 패턴인 접두사, 접미사, 구두점은 어떤 corpus에서도 유사하게 나타나며, 웹 데이터 정규화 파이프라인이 업계에서 표준화되어 있기도 합니다. 실제로 LLaMA와 TigerBot은 독립적으로 학습되었지만 53% vocabulary 중복을 보였습니다. 그러나 95% 이상의 중복은 여전히 의심스러운 수준입니다.

두 번째로 "동일한 Special Token 패턴 = Fine-tuning"이라는 주장에 대한 반론입니다. `<s>`, `</s>`, `<pad>`, `<unk>`는 SentencePiece의 기본값으로 널리 사용됩니다. 기존 도구나 프레임워크와의 호환을 위해 표준 형식을 채택하는 경우가 많습니다. Special token 수가 4-10개 정도로 적어서 우연히 겹칠 확률이 높습니다. Llama-2, Mistral, Solar-Open-100B 모두 `<s>`, `</s>` 형식을 사용하지만, 이것만으로 같은 계열이라고 단정할 수 없습니다. 결론적으로 special token 일치만으로는 fine-tuning을 판단할 수 없고, vocab size와 merge rules를 함께 종합 판단해야 합니다.

세 번째로 "동일한 BPE Merge Rules = Fine-tuning"이라는 주장에 대한 반론입니다. 첫 수백 개 merge는 언어 보편적입니다. 't'+'h'→'th', 'e'+'r'→'er', 'i'+'n'→'in' 같은 패턴은 영어 기반 모델에서 거의 동일하게 나타납니다. 동일한 공개 데이터셋인 Common Crawl이나 Wikipedia를 사용하면 유사한 merge 순서가 나올 수 있습니다. BPE는 결정론적 알고리즘이어서 동일 입력이면 동일 출력이 나옵니다. 따라서 초기 1000개 merge 일치는 큰 의미가 없고, 전체 merge 순서가 일치해야 fine-tuning의 강력한 증거가 됩니다. 특히 후반부 merge가 일치하면 domain-specific하므로 더 강한 증거입니다.

더 강력한 fine-tuning 증거를 살펴보면, vocab size 완전 일치가 가장 신뢰도가 높습니다. 우연의 일치가 거의 불가능하기 때문입니다. Embedding matrix 일치도 마찬가지로 weight까지 같으면 확실합니다. 전체 merge rules 일치는 순서까지 같으면 동일 tokenizer입니다. 95% 이상 vocabulary 중복은 높지만 false positive 가능성이 있습니다. Special tokens 일치는 업계 관행으로 흔하고, tokenizer type 일치는 SentencePiece나 BPE가 표준이어서 신뢰도가 낮습니다.

Solar-Open-100B의 경우를 보면, vocab_size가 196,608로 어떤 모델과도 불일치하므로 fine-tuning 가능성이 낮습니다. Special tokens는 Mixtral과 유사하지만 이는 중립적 증거입니다. 가장 가까운 Qwen2보다도 29% 큰 vocab_size 차이가 있습니다. 반론을 고려하더라도 Solar-Open-100B의 vocab_size가 모든 비교 대상과 크게 다른 점은 여전히 from scratch의 강력한 증거입니다.

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

## Q8: Solar-Open-100B의 행동 분석 결과는? (표절 논란과 공개 검증)

**질문 시각**: 2026-01-04

**답변**:

Solar-Open-100B에 대한 행동 분석을 진행하던 중, 2026년 1월 초 한국에서 발생한 표절 논란과 공개 검증 세션에 대한 정보를 발견했습니다. 이는 from scratch 검증에 매우 중요한 정보입니다.

2026년 1월 1일, Sionic AI CEO 고석현이 LinkedIn과 GitHub에 기술 분석을 게시하며 논란이 시작되었습니다. 그의 주장은 Solar-Open-100B와 Zhipu AI의 GLM-4.5-Air 간에 LayerNorm weight의 cosine similarity가 96.8%라는 것이었습니다. 또한 GLM 스타일 config 코드와 Zhipu AI 라이선스 참조가 발견되었다고 주장했습니다. 그의 결론은 Solar가 fine-tuned 모델이며 국가 AI 프로젝트 규정을 위반했을 가능성이 있다는 것이었습니다.

Zhipu AI의 GLM-4.5-Air와 Solar-Open-100B를 비교해보면 놀라운 유사점이 있습니다. 총 파라미터는 GLM이 106B, Solar가 102.6B입니다. 활성 파라미터는 둘 다 12B입니다. Architecture는 둘 다 MoE이고, Context Length도 둘 다 128K입니다. 다만 GLM-4.5-Air의 상세 config는 비공개입니다.

Upstage는 2026년 1월 2일 서울 강남 사무실에서 공개 검증 세션을 개최하여 대응했습니다. 이 자리에서 training checkpoints, WandB 실험 로그, 중간 산출물, 전체 학습 히스토리를 공개했습니다. Upstage의 주장은 random initialization에서 시작하여 처음부터 학습했으며, 중국 모델 가중치를 재사용하지 않았다는 것이었습니다. 코드 내 중국어 저작권 표시는 실수라고 설명했습니다.

이틀 뒤인 1월 3일, 고석현 CEO는 부분 사과를 발표하며 성급한 판단이었음을 인정했습니다.

그러나 LayerNorm 96.8% 유사도 의혹은 여전히 설명이 필요했습니다. 이에 대해 hyunwoongko가 GitHub에서 독립 검증을 수행했습니다. 그 결과 LayerNorm 96.8% 유사도 주장이 방법론적 오류였음이 밝혀졌습니다. 같은 모델의 다른 레이어 간에도 0.99 수준의 cosine similarity가 나타났습니다. 이는 LayerNorm weight가 1.0으로 초기화되어 방향적 일관성을 유지하기 때문입니다. 평균 오프셋을 제거한 centered cosine 분석을 하면 모델 간 유사도가 거의 0으로 하락했습니다. Solar가 GLM보다 Phi-3.5-MoE에 더 가깝다는 증거도 없었습니다. 결론적으로 LayerNorm 비교는 모델 기원 판별에 부적합하며, 원래 주장은 초기화 편향에 의한 false positive였습니다.

행동 분석의 한계도 있습니다. Knowledge cutoff가 공식적으로 공개되지 않아 직접 비교가 어렵습니다. Safety alignment 정보도 미공개여서 refusal pattern 분석이 제한적입니다. 직접 실행 환경이 없어 출력 스타일 비교도 제한적입니다. GLM-4.5-Air config가 미공개여서 LayerNorm 유사도를 직접 확인할 수도 없습니다.

종합하면, Upstage가 training logs와 checkpoints 등 증거를 공개했고, 외부 전문가를 초청한 공개 검증을 진행했으며, 고석현 CEO가 부분 사과했습니다. 독립 검증에서 LayerNorm 유사도 주장이 방법론적 오류로 밝혀졌습니다. 행동 분석과 독립 검증을 종합하면, from scratch 주장은 신뢰할 수 있습니다.

---

## Q9: HyperCLOVAX-SEED-Think-32B는 from scratch인가요?

**질문 시각**: 2026-01-05

**답변**:

NAVER Cloud의 HyperCLOVAX-SEED-Think-32B를 분석한 결과, 완전한 from scratch라기보다는 부분적 from scratch와 컴포넌트 재사용이 혼합된 구조로 확인되었습니다.

이 모델은 VLM(Vision-Language Model)으로 세 가지 컴포넌트로 구성됩니다. Vision Encoder, Text Decoder, 그리고 Projector입니다.

가장 명확한 발견은 Vision Encoder가 Qwen2.5 ViT를 재사용한다는 것입니다. config.json에 `"model_type": "qwen2_5_vl"`이 명시되어 있습니다. 따라서 Vision 부분은 from scratch가 아닙니다.

Tokenizer에 대해서는 흥미로운 발견이 있었습니다. HyperCLOVAX-SEED의 vocab_size는 128,256개입니다. Llama 3/3.1은 128,000개로 256 토큰 차이가 납니다. 처음에는 Llama 3 tokenizer를 재사용했다고 생각할 수 있지만, 정확히 일치하지 않습니다. Trillion-7B도 128,256개로 동일한 vocab 설계를 사용합니다. Trillion-7B 논문에 따르면 이 128,256 vocab은 약 100k 영어와 약 24.5k 한국어로 구성된 한국어 최적화 설계입니다. 단순히 Llama 3 tokenizer를 재사용한 것이 아닐 가능성이 있습니다. 다만 HyperCLOVA X 논문에서 언급된 100k vocab에서 SEED 버전의 128k vocab으로 변경된 이유가 공식 문서화되지 않아 완전한 결론은 어렵습니다.

Text Decoder의 architecture를 보면 고유한 요소가 있습니다. model_type이 "hyperclovax"로 독자적입니다. hidden_size는 5,120으로 Llama 3.1 70B의 약 8,192나 Qwen2.5-72B의 12,288과 다릅니다. num_layers는 72개로 Llama와 Qwen의 80개와 다릅니다. num_heads는 40개로 역시 다릅니다. 특히 주목할 점은 rope_theta가 50,000,000이라는 것입니다. Llama 3.1은 500,000, Qwen2.5는 1,000,000인데, HyperCLOVAX는 50M으로 다른 모델에서 볼 수 없는 고유값입니다.

결론적으로 완전한 from scratch라고 보기 어렵습니다. Vision Encoder가 Qwen2.5 ViT를 그대로 사용한다는 점이 config.json에 명시되어 있어서, VLM에서 Vision 부분은 확실히 재사용했습니다. Tokenizer에 대해서는 Llama 3와 256 토큰 차이가 있어 단순 재사용이 아닐 수 있지만, 추가 검증이 필요합니다. Text Decoder는 architecture가 고유하지만 이것만으로 from scratch를 확정하기 어렵습니다.

---

## Q10: vocab_size 일치가 왜 중요한가요? 우연히 같은 숫자가 나올 수도 있지 않나요?

**질문 시각**: 2026-01-05

**답변**:

vocab_size가 우연히 일치할 가능성에 대해 분석해 보겠습니다.

실제 LLM들이 사용하는 vocab_size 분포를 보면, Llama 2와 Mixtral은 32,000개를 사용하고, GPT-2나 RoBERTa는 50,000에서 65,000 사이를 사용합니다. Qwen이나 DeepSeek은 100,000에서 150,000 사이이고, Solar나 K-EXAONE은 150,000에서 200,000 사이입니다. 이론적으로 가능한 값은 수천에서 수십만까지 연속적으로 선택할 수 있습니다.

vocab_size를 결정하는 요인은 학습 corpus의 언어 분포, character_coverage 설정, 목표 vocab_size 설정, 그리고 BPE나 SentencePiece 알고리즘의 결과입니다. 독립적으로 학습하면 다른 corpus를 사용하므로 다른 token 빈도 분포가 나오고, 다른 설정을 사용하면 다른 최종 vocab_size가 나옵니다. 정확히 같은 숫자가 나올 확률은 극히 낮습니다.

"비슷한 숫자"와 "정확히 같은 숫자"는 의미가 다릅니다. 128,000과 128,256처럼 256 차이가 나면 독립 설계 가능성이 있습니다. 152,064와 153,600처럼 1,536 차이가 나도 마찬가지입니다. 하지만 128,256과 128,256처럼 정확히 0 차이면 동일 tokenizer 사용을 강력히 의심해야 합니다.

이 프로젝트에서 검증한 5개 모델을 보면, Solar-Open-100B는 196,608개로 일치하는 모델이 없습니다. A.X-K1은 163,840개, K-EXAONE은 153,600개, VAETKI는 137,216개로 모두 일치하는 모델이 없습니다. HyperCLOVAX-SEED만 128,256개로 Trillion-7B와 일치합니다. 5개 중 4개가 모든 기존 모델과 불일치하는 것은 독립 설계 시 불일치가 정상적인 결과임을 보여줍니다.

물론 vocab_size 단독으로는 판단이 불충분합니다. 실제 vocabulary 내용을 비교하여 token 중복률을 계산해야 하고, BPE merge rules의 순서까지 같은지 확인해야 합니다. 고유한 special token 패턴도 살펴봐야 합니다.

결론적으로 vocab_size가 정확히 일치하면 우연의 일치 확률은 0.01% 미만입니다. 연속 변수에서 정확히 같은 값이 나올 확률이 그만큼 낮기 때문입니다. 이는 "같은 tokenizer 사용" 또는 "의도적으로 맞춤"의 강력한 증거입니다. 반면 비슷하지만 다르면 독립 설계 가능성이 열려 있고, 추가 검증이 필요합니다.

이 프로젝트에서 vocab_size를 중요하게 보는 이유는, 정확히 일치하는 경우가 없어야 from scratch의 기본 조건이 충족되기 때문입니다. HyperCLOVAX-SEED의 경우 Llama 3와 256 차이가 나서 "단순 재사용이 아닐 수 있다"고 재해석한 것도 이 논리입니다.

---

## Q11: 이 프로젝트의 검증 방법론에 대한 비판과 학술 연구 결과는?

**질문 시각**: 2026-01-05

**답변**:

"vocab_size나 config dimension 비교만으로는 불충분하고, 실제 토큰 중복률이나 weight 값 비교가 더 중요하다"는 의견이 제기되었습니다. 이에 대해 학술 연구를 조사한 결과를 공유합니다.

학술적으로 가장 신뢰할 수 있는 검증 방법은 무엇일까요? arXiv:2502.00706v1에 발표된 "Model Provenance Testing for Large Language Models" 논문에 따르면, 600개 이상의 모델에서 검증된 결과 출력 행동 기반 통계 분석이 가장 신뢰할 수 있는 방법입니다. 동일 프롬프트에 대한 출력 토큰 일치율을 통계 분석하는 black-box output similarity 방법이 90-95% precision과 80-90% recall을 달성했습니다. Config 비교는 불충분한데, fine-tuning은 architecture를 보존하기 때문입니다. Weight 비교는 유효하지만 white-box 접근이 필요합니다.

Config나 dimension 비교에는 한계가 있습니다. config.json 비교의 한계는 fine-tuning이 architecture를 보존하므로 동일한 dimension을 가진 모델이 파생작일 수 있다는 점입니다. vocab_size 비교의 경우, fine-tuning 시 tokenizer를 그대로 사용하므로 vocab_size 일치가 오히려 파생 증거가 될 수 있습니다.

그렇다면 현재 프로젝트 방법론의 유효성은 어떨까요? 현재 프로젝트의 로직은 "vocab_size가 불일치하면 from scratch 지지"입니다. vocab_size가 다르면 tokenizer 재학습이 필요하므로 from scratch의 강력한 증거가 됩니다. 이 로직은 유효합니다. 반면 vocab_size가 같으면 추가 검증이 필요한데, token 중복률과 merge rules를 분석해야 합니다. 따라서 vocab_size 불일치를 기반으로 한 Solar, A.X-K1, VAETKI, K-EXAONE의 판정은 여전히 유효합니다. HyperCLOVAX처럼 vocab_size가 유사한 경우에는 실제 토큰 중복률 분석이 추가로 필요합니다.

Yi-Llama 논란 사례도 참고할 만합니다. 01.AI의 Yi-34B 모델이 Meta Llama에서 파생되었다는 의혹이 있었습니다. Architecture가 Llama와 동일한 구조였고, tensor names도 Llama 형식을 그대로 사용했습니다. 01.AI는 이를 "oversight(실수)"라고 인정했습니다. 하지만 weight 복사 증거는 없었고 독립 학습 주장을 유지했습니다. EleutherAI의 분석에 따르면, Yi는 독립적으로 학습되었으며 Llama architecture 채택은 업계 표준 관행입니다.

방법론 개선 방향을 정리하면, vocab_size 비교에 실제 토큰 중복률 계산을 추가하고, config dimension 비교에 출력 행동 유사도 분석을 추가하면 됩니다. Architecture 비교는 유효한데, architecture가 다르면 weight 재사용이 불가능하기 때문입니다.

비판의 타당성을 평가하면, "config dimension 비교만으로 불충분"하다는 주장은 부분적으로 맞습니다. 같은 architecture에서 fine-tuning이 가능하기 때문입니다. "실제 토큰 중복률이 더 중요"하다는 주장은 맞는데, 특히 vocab_size가 유사할 때 그렇습니다. "weight 비교가 더 중요"하다는 주장은 부분적으로 맞지만, 학술 연구에서는 output behavior가 더 신뢰성 있는 것으로 나타났습니다.

현재 프로젝트 판정의 유효성을 정리하면, vocab_size가 명확히 다른 모델들인 Solar, A.X-K1, VAETKI, K-EXAONE의 판정은 유효합니다. vocab_size가 유사한 HyperCLOVAX는 추가 검증을 권장합니다.

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

<!-- TUTORIAL_MARKER: 새로운 Q&A는 이 마커 위에 자동 추가됩니다 -->
