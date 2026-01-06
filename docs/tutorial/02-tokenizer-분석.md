# Tokenizer 분석

Tokenizer 기반 검증 방법과 실제 분석 결과를 다룹니다.

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

<!-- SECTION_MARKER: 새로운 Tokenizer 분석 Q&A는 이 마커 위에 추가됩니다 -->
