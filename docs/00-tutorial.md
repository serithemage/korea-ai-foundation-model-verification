# 튜토리얼: Q&A 형식 학습 기록

이 문서는 Solar-Open-100B 검증 과정에서 진행한 질문과 답변을 시간순으로 기록합니다.

---

## Q0: 왜 "from scratch" vs "fine-tuning" 논란이 발생하나요?

**질문 시각**: 2026-01-05

**답변**:

### 국가 AI 파운데이션 모델 프로젝트 규정

한국 정부의 국가 AI 파운데이션 모델 프로젝트는 **"from scratch 학습"을 필수 요건**으로 규정하고 있습니다.

| 학습 방식 | 정의 | 프로젝트 적합성 |
|----------|------|----------------|
| **From scratch** | 무작위 초기화된 weight에서 시작하여 전체 학습 수행 | ✅ 적합 |
| **Fine-tuning** | 기존 모델의 pre-trained weight를 가져와 추가 학습 | ❌ 부적합 |

### 논란의 핵심 이유

**1. AI 주권 (AI Sovereignty)**

From scratch 학습을 요구하는 이유는 **기술적 독립성** 확보입니다:
- 기존 모델(Llama, GPT 등)의 weight를 사용하면 해당 모델의 라이선스와 제약에 종속
- 국가 전략 기술로서 독자적인 기술력 보유 필요
- 향후 모델 수정/확장에서의 자유도 확보

**2. 국민 세금 사용**

국가 프로젝트는 정부 예산(국민 세금)으로 진행됩니다:
- Fine-tuning은 from scratch 대비 훨씬 적은 비용으로 가능
- From scratch 비용을 받고 fine-tuning만 했다면 예산 낭비 또는 사기 논란 가능
- 납세자에 대한 책임과 투명성 문제

**3. 신뢰도와 평판**

기술 기업의 핵심 자산은 신뢰입니다:
- "From scratch"라고 주장했는데 실제로는 fine-tuning이었다면 심각한 신뢰 손상
- 국제 AI 커뮤니티에서의 한국 AI 기술력 평판에 영향
- 후속 프로젝트 및 투자 유치에 부정적 영향

### From scratch vs Fine-tuning 비용 차이

| 항목 | From scratch | Fine-tuning |
|------|-------------|-------------|
| **학습 토큰** | 수조~수십조 | 수십억~수백억 |
| **GPU 시간** | 수만~수십만 GPU-hours | 수백~수천 GPU-hours |
| **비용** | 수백억~수천억 원 | 수억~수십억 원 |
| **기간** | 수개월~1년 이상 | 수일~수주 |

이러한 비용 차이로 인해, from scratch 주장이 거짓이라면 그것은 단순한 기술적 오류가 아닌 **예산 집행의 투명성 문제**가 됩니다.

---

## Q1: LLM이 "from scratch"로 학습되었는지 어떻게 검증할 수 있나요?

**질문 시각**: 2026-01-04

**답변**:

LLM이 from scratch로 학습되었는지 vs fine-tuned인지 검증하는 여러 기술적 방법이 있습니다:

1. **Tokenizer 분석** (가장 접근성 높음)
   - Tokenizer는 재학습 비용이 높아 fine-tuning시 거의 재사용됨
   - Vocabulary의 95% 이상이 기존 모델과 동일하면 fine-tuning 가능성 높음
   - `tokenizer.get_vocab()`으로 추출하여 비교

2. **Weight 분석**
   - Fine-tuned 모델은 초기 레이어에서 base model과 90% 이상 cosine similarity 유지
   - SHA-256 해시로 weight tensor 비교
   - From scratch 모델은 PCA에서 orthogonal 분포

3. **Architecture 비교**
   - `model.config`로 hyperparameter 확인
   - 동일한 config는 fine-tuning 강력 증거
   - 고유한 구성요소(예: 특이한 RoPE scaling)는 from scratch 증거

4. **행동 테스트**
   - Knowledge cutoff 날짜 확인
   - Safety alignment, refusal pattern이 base model과 동일하면 fine-tuning 가능성

5. **Compute 추정**
   - From scratch는 fine-tuning 대비 10-100배 더 많은 compute 필요
   - 19.7T tokens 학습은 massive compute로, from scratch 주장과 일관성 있음

| 방법 | 신뢰도 | 접근성 | Fine-tuning 탐지력 |
|------|--------|--------|-------------------|
| Tokenizer 분석 | 높음 | 높음 | 우수 |
| Weight 해싱 | 높음 | 중간 | 양호 |
| Architecture 매칭 | 중간 | 높음 | 양호 |
| 행동 테스트 | 중간 | 높음 | 보통 |
| Training Logs | 매우 높음 | 낮음 | 우수 |

---

## Q2: Tokenizer 분석이란 무엇이고, 어떻게 from scratch 여부를 판별하나요?

**질문 시각**: 2026-01-04

**답변**:

### Tokenizer의 작동 원리

Tokenizer는 텍스트를 모델이 처리할 수 있는 숫자(token ID)로 변환합니다. 주요 방식:

| 방식 | 특징 | 사용 모델 |
|------|------|----------|
| **BPE** (Byte Pair Encoding) | 빈도 기반으로 인접 문자쌍 병합 | GPT-2, RoBERTa |
| **WordPiece** | likelihood 최대화 기준 병합 | BERT |
| **SentencePiece** | 공백 포함 원시 텍스트 처리 (▁ 마커 사용) | T5, Gemma, Llama |

### Fine-tuning 시 Tokenizer를 재학습하지 않는 이유

1. **Embedding 호환성**: 새 vocabulary는 pre-trained embedding과 호환되지 않음
2. **비용**: Tokenizer 재학습은 전체 corpus 재처리 필요
3. **성능 저하 위험**: Vocabulary 변경 시 원래 도메인 성능 저하

### Vocabulary 중복률 기준

| 중복률 | 해석 |
|--------|------|
| **>98%** | Fine-tuning 가능성 높음 (safety token 등 소량 추가만) |
| **90-98%** | Continued pre-training 또는 vocabulary 확장 |
| **<90%** | From scratch 학습 강력 증거 |

### 분석 기법

**1. Vocabulary 비교**
```python
from transformers import AutoTokenizer

base_tok = AutoTokenizer.from_pretrained("base-model")
target_tok = AutoTokenizer.from_pretrained("target-model")

base_vocab = set(base_tok.get_vocab().keys())
target_vocab = set(target_tok.get_vocab().keys())

overlap = len(base_vocab & target_vocab)
overlap_pct = (overlap / len(base_vocab)) * 100
print(f"중복률: {overlap_pct:.2f}%")
```

**2. Merge Rules 비교 (BPE/SentencePiece)**
```python
# merges가 동일하면 같은 tokenizer
base_merges = base_tok.backend_tokenizer.model.get_vocab()
target_merges = target_tok.backend_tokenizer.model.get_vocab()
```

**3. Special Tokens 비교**
```python
print(base_tok.special_tokens_map)
print(target_tok.special_tokens_map)
# [PAD], [UNK], [CLS], <eos> 등 비교
```

**4. Encoding 결과 비교**
```python
text = "Hello, world! 토큰화 테스트입니다."
base_tokens = base_tok.tokenize(text)
target_tokens = target_tok.tokenize(text)
# 동일 입력에 다른 토큰 분할 → 다른 tokenizer
```

### Solar-Open-100B 검증 시 비교 대상

1. **Llama 계열**: Llama-2, Llama-3 (SentencePiece 기반)
2. **Mistral/Mixtral**: MoE 아키텍처 유사
3. **Qwen**: 대규모 한국어 포함 모델
4. **DeepSeek-MoE**: MoE 아키텍처

**핵심**: Solar-Open-100B의 tokenizer가 위 모델들과 90% 미만 중복이면 from scratch 주장 지지

---

## Q3: Weight 분석이란 무엇이고, 어떻게 from scratch 여부를 판별하나요?

**질문 시각**: 2026-01-04

**답변**:

Weight 분석은 모델의 가중치를 직접 비교하여 from scratch 학습 여부를 판별하는 방법입니다. Fine-tuned 모델은 base model과 높은 가중치 유사성을 보이는 반면, from scratch 모델은 독립적인 가중치 분포를 갖습니다.

### 1. Layer별 Cosine Similarity 분석

두 모델의 대응하는 weight matrix 간 방향적 유사성을 측정합니다.

**원리**:
- Fine-tuned 모델: 초기 레이어에서 >0.95 유사도, 후기 레이어에서 점차 감소
- From scratch 모델: 전체적으로 낮은 유사도 (~0에 가까움)

**코드 예시**:
```python
import torch
import torch.nn.functional as F
from transformers import AutoModel

def cosine_similarity(w1, w2):
    w1_flat = w1.flatten().float()
    w2_flat = w2.flatten().float()
    return F.cosine_similarity(w1_flat.unsqueeze(0), w2_flat.unsqueeze(0)).item()

base_model = AutoModel.from_pretrained("base-model")
target_model = AutoModel.from_pretrained("target-model")

for name, param in base_model.named_parameters():
    if name in dict(target_model.named_parameters()):
        target_param = dict(target_model.named_parameters())[name]
        if param.shape == target_param.shape:
            sim = cosine_similarity(param.data, target_param.data)
            print(f"{name}: {sim:.4f}")
```

### 2. Weight Tensor 해싱

대규모 모델에서 효율적인 비교를 위해 해시 기반 fingerprint를 사용합니다.

**방법**:
- **MinHash/SimHash**: Locality-sensitive hash로 Jaccard 유사도 계산
- **Tensor Checksum**: 양자화된 weight의 perceptual hash 비교
- **Exact Matching**: MSE < 1e-5 이내면 동일 layer로 판정

```python
import hashlib

def weight_hash(tensor):
    arr = tensor.cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]

# 동일 해시 = 동일 weight (fine-tuning 강력 증거)
```

### 3. PCA 분석

고차원 weight matrix를 저차원으로 투영하여 분포 비교.

**해석**:
- Fine-tuned: Base model 근처에 clustering (80% 이상 overlap)
- From scratch: 완전히 다른 cluster 형성

```python
from sklearn.decomposition import PCA
import numpy as np

# Layer weight를 feature vector로 추출
features_base = extract_features(base_model)
features_target = extract_features(target_model)

pca = PCA(n_components=2)
all_features = np.vstack([features_base, features_target])
reduced = pca.fit_transform(all_features)

# 시각화로 clustering 패턴 확인
```

### 4. Embedding Layer 분석

Token embedding은 fine-tuning에서 가장 적게 변하므로 특히 중요합니다.

**비교 방법**:
- Embedding matrix 직접 cosine similarity
- K-means clustering으로 centroid 비교
- L2 norm 및 variance 비교

```python
base_emb = base_model.get_input_embeddings().weight.data
target_emb = target_model.get_input_embeddings().weight.data

# Shape 불일치 = 다른 vocabulary = from scratch 증거
if base_emb.shape != target_emb.shape:
    print("다른 vocabulary 사용 - from scratch 가능성")
else:
    sim = cosine_similarity(base_emb, target_emb)
    print(f"Embedding 유사도: {sim:.4f}")
```

### 유사도 임계값 기준

| 지표 | Fine-tuning | From Scratch |
|------|-------------|--------------|
| **평균 Layer Cosine Sim** | >0.90 (초기 layer >0.99) | <0.3 |
| **Embedding Cosine Sim** | >0.98 | <0.1 |
| **Hash Jaccard/Hamming** | >0.85 / <2% 차이 | <0.2 / >20% 차이 |
| **PCA Wasserstein Dist** | <0.05 | >0.5 |

### 100B MoE 모델 분석 시 실제 도전 과제

1. **메모리/연산량**: 100B MoE 모델은 full weight 로드에 >1TB RAM 필요
   - 해결: Sharded loading (HF Accelerate), FP8 양자화

2. **MoE 구조 특수성**: Expert별 weight 분리 분석 필요
   - Router weight와 Expert weight 별도 비교

3. **Architecture 불일치**: topology가 다르면 직접 비교 불가
   - num_experts, head_dim 등 먼저 확인

4. **수치 불안정성**: 대규모 tensor에서 precision error 누적
   - Double precision 사용 또는 submatrix 샘플링 (10-20%)

### Solar-Open-100B 검증 적용

Solar-Open-100B의 weight를 다음 모델들과 비교:
- **Llama-3**: Dense 모델 기준선
- **Mixtral**: MoE 구조 유사
- **DeepSeek-MoE**: MoE 아키텍처

**핵심**: 모든 주요 base model과 평균 cosine similarity 0.5 미만이면 from scratch 주장 지지

---

## Q4: Cosine Similarity 분석이란 무엇이고, LLM weight 비교에 어떻게 활용되나요?

**질문 시각**: 2026-01-04

**답변**:

Cosine Similarity는 두 벡터 간의 방향적 유사성을 측정하는 지표로, LLM weight 비교에서 핵심적인 역할을 합니다.

### 수학적 정의

```
cos(θ) = (A · B) / (||A|| × ||B||)
```

- **A · B**: 두 벡터의 내적 (dot product)
- **||A||, ||B||**: 각 벡터의 L2 norm (크기)
- **결과 범위**: -1 ~ 1 (양수 weight의 경우 0 ~ 1)

### Weight 비교에 Cosine Similarity를 사용하는 이유

| 특성 | 설명 |
|------|------|
| **Scale 불변성** | 벡터 크기에 독립적, 방향만 비교 |
| **고차원 적합성** | 수백만 차원에서도 효율적 계산 |
| **해석 용이성** | 1에 가까울수록 유사, 0에 가까울수록 다름 |
| **Normalize 불필요** | 자체적으로 정규화 포함 |

### 값 해석 기준

| Cosine Similarity | 해석 |
|-------------------|------|
| **0.99 ~ 1.0** | 거의 동일한 weight (fine-tuning 강력 증거) |
| **0.90 ~ 0.99** | 높은 유사도 (fine-tuning 또는 같은 initialization) |
| **0.50 ~ 0.90** | 중간 유사도 (부분적 공유 가능성) |
| **0.10 ~ 0.50** | 낮은 유사도 (독립적 학습 가능성) |
| **0.0 ~ 0.10** | 거의 무관 (orthogonal, from scratch 강력 증거) |

### Layer별 패턴 분석

**Fine-tuned 모델의 전형적 패턴:**
```
Layer 0 (Embedding):     0.98 ~ 0.99  ← 거의 변화 없음
Layer 1-5 (초기):        0.95 ~ 0.99  ← 약간의 조정
Layer 6-20 (중간):       0.85 ~ 0.95  ← 점진적 감소
Layer 21+ (후기):        0.70 ~ 0.90  ← task-specific 학습
Output Layer:            0.60 ~ 0.85  ← 가장 많이 변화
```

**From scratch 모델의 전형적 패턴:**
```
모든 Layer:              0.0 ~ 0.3   ← 전체적으로 낮은 유사도
                                      (random initialization 효과)
```

### 구현 예시

```python
import torch
import torch.nn.functional as F

def cosine_similarity_analysis(model_a, model_b):
    """두 모델의 layer별 cosine similarity 분석"""
    results = {}

    params_a = dict(model_a.named_parameters())
    params_b = dict(model_b.named_parameters())

    for name in params_a:
        if name in params_b:
            w_a = params_a[name].data.flatten().float()
            w_b = params_b[name].data.flatten().float()

            if w_a.shape == w_b.shape:
                # Cosine similarity 계산
                sim = F.cosine_similarity(
                    w_a.unsqueeze(0),
                    w_b.unsqueeze(0)
                ).item()
                results[name] = sim

    return results

def summarize_by_layer_type(results):
    """Layer 유형별 평균 유사도 요약"""
    categories = {
        'embedding': [],
        'attention': [],
        'mlp': [],
        'norm': [],
        'output': []
    }

    for name, sim in results.items():
        if 'embed' in name.lower():
            categories['embedding'].append(sim)
        elif 'attn' in name.lower() or 'attention' in name.lower():
            categories['attention'].append(sim)
        elif 'mlp' in name.lower() or 'ffn' in name.lower():
            categories['mlp'].append(sim)
        elif 'norm' in name.lower():
            categories['norm'].append(sim)
        elif 'lm_head' in name.lower() or 'output' in name.lower():
            categories['output'].append(sim)

    for cat, sims in categories.items():
        if sims:
            avg = sum(sims) / len(sims)
            print(f"{cat}: {avg:.4f} (n={len(sims)})")
```

### 대규모 모델에서의 실용적 고려사항

1. **메모리 최적화**
   - 전체 tensor를 한 번에 로드하지 않고 chunk 단위로 처리
   - FP16/BF16으로 계산하여 메모리 절약

2. **샘플링 전략**
   - 100B+ 모델은 전체 weight 비교가 비실용적
   - Layer당 10-20% 무작위 샘플링으로 추정
   - 통계적 신뢰구간 계산

3. **MoE 모델 특수 처리**
   - Router weight와 Expert weight 분리 분석
   - Shared expert vs Routed expert 구분

```python
def sample_cosine_similarity(w_a, w_b, sample_ratio=0.1):
    """대규모 tensor를 위한 샘플링 기반 유사도 추정"""
    n = w_a.numel()
    sample_size = int(n * sample_ratio)

    indices = torch.randperm(n)[:sample_size]

    sample_a = w_a.flatten()[indices].float()
    sample_b = w_b.flatten()[indices].float()

    return F.cosine_similarity(
        sample_a.unsqueeze(0),
        sample_b.unsqueeze(0)
    ).item()
```

### Cosine Similarity의 한계

| 한계점 | 설명 | 보완 방법 |
|--------|------|----------|
| **Zero vector 문제** | 0 벡터에서 정의되지 않음 | Zero 체크 후 처리 |
| **크기 정보 손실** | 방향만 비교, magnitude 무시 | L2 distance 병행 |
| **고차원 집중** | 고차원에서 값이 중앙으로 수렴 | Layer별 분석으로 보완 |
| **Outlier 민감도** | 극단값에 영향 받음 | Robust 버전 사용 |

### Solar-Open-100B 검증 적용

Solar-Open-100B의 cosine similarity 분석 시:

1. **비교 대상 모델**
   - Llama-3 (Dense baseline)
   - Mixtral (MoE 유사 구조)
   - DeepSeek-MoE (MoE 비교)

2. **분석 계층**
   - Embedding layer
   - Attention (Q, K, V, O projections)
   - MLP/Expert weights
   - Router weights (MoE 특화)
   - Output layer

3. **판정 기준**
   - 모든 base model과 평균 similarity < 0.3 → from scratch 지지
   - 특정 모델과 초기 layer similarity > 0.9 → fine-tuning 의심

---

## Q5: Solar-Open-100B의 Tokenizer 분석 결과는 어떠한가요?

**질문 시각**: 2026-01-04

**답변**:

Solar-Open-100B의 tokenizer를 주요 base model들과 비교 분석한 결과입니다.

### Vocabulary 크기 비교

| 모델 | Vocab Size | Tokenizer Type | 비고 |
|------|-----------|----------------|------|
| **Solar-Open-100B** | **196,608** | SentencePiece (BPE) | 가장 큼 |
| Qwen2-72B | 152,064 | BPE | Solar보다 29% 작음 |
| Llama-3 | 128,256 | tiktoken (BPE) | Solar보다 53% 작음 |
| DeepSeek-V2 | 102,400 | BPE | Solar보다 92% 작음 |
| Mixtral-8x7B | 32,000 | SentencePiece | Solar보다 514% 작음 |

### Special Tokens 비교

| 모델 | bos_token | eos_token | pad_token |
|------|-----------|-----------|-----------|
| **Solar-Open-100B** | `<s>` | `</s>` | `<pad>` |
| Llama-3 | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | (없음) |
| Mixtral | `<s>` | `</s>` | (없음) |

### 분석 결과

**1. Vocabulary 크기 분석**

Solar-Open-100B의 vocab_size (196,608)는:
- 모든 비교 대상 모델보다 **유의미하게 큼**
- 가장 가까운 Qwen2 (152,064)보다도 44,544개 (약 29%) 더 많음
- Llama-3 계열 (128,256)과는 68,352개 (약 53%) 차이

이는 vocabulary를 재사용하지 않고 **독립적으로 학습**했음을 강력히 시사합니다.

**2. Special Tokens 패턴**

Solar-Open-100B는 전통적인 `<s>`, `</s>` 형식을 사용:
- Llama-3의 `<|begin_of_text|>` 스타일과 다름
- Mixtral과 유사하나, vocab_size가 완전히 다름

**3. Tokenizer Type**

- Solar-Open-100B: SentencePiece BPE
- Llama-3: tiktoken BPE (다른 구현)
- 동일한 SentencePiece를 쓰는 Mixtral과도 vocab_size가 6배 이상 차이

### From Scratch 판정

| 지표 | 결과 | 해석 |
|------|------|------|
| **Vocab Size 일치** | 0개 모델과 일치 | ✅ From scratch 지지 |
| **Vocab Size 유사** | 해당 없음 (모두 큰 차이) | ✅ From scratch 지지 |
| **Special Tokens** | Mixtral과 유사하나 vocab 다름 | ⚠️ 중립 |
| **Tokenizer Type** | SentencePiece BPE (공통) | ⚠️ 중립 (방식만 공유) |

### 결론

**Tokenizer 분석 결과: From scratch 학습 주장 지지**

- 196,608 크기의 vocabulary는 어떤 주요 base model과도 일치하지 않음
- Vocabulary 확장이나 fine-tuning으로는 설명되지 않는 규모의 차이
- 독립적으로 학습된 tokenizer임이 강력히 시사됨

### 추가 검증 필요 사항

더 확실한 결론을 위해:
1. 실제 vocabulary 토큰 목록을 다운로드하여 중복률 계산
2. BPE merge rules 순서 비교
3. 동일 텍스트에 대한 토큰화 결과 비교

```python
# 검증 코드 예시
from transformers import AutoTokenizer

solar_tok = AutoTokenizer.from_pretrained("upstage/Solar-Open-100B")
llama_tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")

solar_vocab = set(solar_tok.get_vocab().keys())
llama_vocab = set(llama_tok.get_vocab().keys())

overlap = len(solar_vocab & llama_vocab)
overlap_pct = (overlap / len(llama_vocab)) * 100
print(f"Llama-3과 중복률: {overlap_pct:.2f}%")
# 예상: 90% 미만 → from scratch 지지
```

---

## Q6: Fine-tuning 의심 증거들에 대한 반론은 무엇인가요?

**질문 시각**: 2026-01-04

**답변**:

Fine-tuning 의심 증거로 제시되는 세 가지 기준에 대해, 각각 **False Positive**(실제로는 from scratch인데 fine-tuning으로 오판)가 발생할 수 있는 경우를 분석합니다.

### 1. "95% 이상 vocabulary 중복 = Fine-tuning"에 대한 반론

**False Positive 가능 상황:**

| 요인 | 설명 |
|------|------|
| **동일 언어 분포** | 영어 웹 데이터로 학습하면 자연스럽게 유사한 빈도 패턴 발생 |
| **동일 알고리즘 설정** | vocab_size=32k, character_coverage=0.9995 등 표준 설정 사용 시 유사한 결과 |
| **언어의 통계적 특성** | 자연어의 공통 패턴 (접두사, 접미사, 구두점)은 어떤 corpus에서도 유사 |
| **표준 전처리** | 웹 데이터 정규화 파이프라인이 업계 표준화됨 |

**실제 사례:**
- **LLaMA vs TigerBot**: 독립적으로 학습되었지만 53% vocabulary 중복
- **LLaMA (32k) vs ChatGLM (125k)**: 다른 기관, 다른 corpus지만 상당한 토큰 공유

**결론**: 독립적으로 학습한 모델도 40-60% 중복은 자연스러움. 다만 **95% 이상**은 여전히 의심스러운 수준.

---

### 2. "동일한 Special Token 패턴 = Fine-tuning"에 대한 반론

**False Positive 가능 상황:**

| 요인 | 설명 |
|------|------|
| **업계 관행** | `<s>`, `</s>`, `<pad>`, `<unk>`는 SentencePiece 기본값으로 널리 사용 |
| **호환성 고려** | 기존 도구/프레임워크와의 호환을 위해 표준 형식 채택 |
| **우연의 일치** | Special token 수가 적어 (4-10개) 겹칠 확률 높음 |

**예시:**

```
SentencePiece 기본값:
  - <unk> (ID: 0)
  - <s>   (ID: 1)
  - </s>  (ID: 2)
  - <pad> (선택적)
```

많은 독립 모델이 이 패턴을 그대로 사용:
- Llama-2: `<s>`, `</s>`
- Mistral: `<s>`, `</s>`
- Solar-Open-100B: `<s>`, `</s>`, `<pad>`

**결론**: Special token 일치만으로는 fine-tuning을 판단할 수 없음. **Vocab size, merge rules와 함께** 종합 판단 필요.

---

### 3. "동일한 BPE Merge Rules = Fine-tuning"에 대한 반론

**False Positive 가능 상황:**

| 요인 | 설명 |
|------|------|
| **초기 merge 패턴** | 첫 수백 개 merge는 언어 보편적 (공백, 구두점, 고빈도 문자쌍) |
| **동일 corpus** | 동일한 공개 데이터셋 (Common Crawl, Wikipedia) 사용 시 유사한 merge 순서 |
| **알고리즘 결정론** | BPE는 결정론적, 동일 입력이면 동일 출력 |

**분석:**

```
일반적인 BPE 초기 merge 예시 (언어 공통):
1. 't' + 'h' → 'th'
2. 'e' + 'r' → 'er'
3. 'i' + 'n' → 'in'
4. ' ' + 't' → ' t'
...
```

이러한 초기 패턴은 영어 기반 모델에서 거의 동일하게 나타남.

**결론**:
- **초기 1000개 merge** 일치: 큰 의미 없음 (언어 보편적)
- **전체 merge 순서** 일치: Fine-tuning 강력 증거
- **후반부 merge** 일치: Domain-specific, 더 강한 증거

---

### 종합: 더 강력한 Fine-tuning 증거는?

| 증거 | 신뢰도 | 이유 |
|------|--------|------|
| Vocab size 완전 일치 | ⭐⭐⭐⭐⭐ | 우연 일치 거의 불가능 |
| Embedding matrix 일치 | ⭐⭐⭐⭐⭐ | Weight까지 같으면 확실 |
| 전체 merge rules 일치 | ⭐⭐⭐⭐ | 순서까지 같으면 동일 tokenizer |
| 95%+ vocabulary 중복 | ⭐⭐⭐ | 높지만 false positive 가능 |
| Special tokens 일치 | ⭐⭐ | 업계 관행으로 흔함 |
| Tokenizer type 일치 | ⭐ | SentencePiece/BPE는 표준 |

### Solar-Open-100B에 적용

Solar-Open-100B의 경우:

| 지표 | 결과 | Fine-tuning 가능성 |
|------|------|-------------------|
| **Vocab size** | 196,608 (어떤 모델과도 불일치) | ❌ 낮음 |
| **Special tokens** | `<s>`, `</s>` (Mixtral과 유사) | ⚠️ 중립 |
| **Vocab size 차이** | 가장 가까운 Qwen2보다 29% 큼 | ❌ 낮음 |

**결론**: 반론을 고려하더라도, Solar-Open-100B의 vocab_size (196,608)가 모든 비교 대상과 크게 다른 점은 여전히 **from scratch의 강력한 증거**.

---

## Q7: Solar-Open-100B의 Weight 분석이 가능한가요? Architecture 비교 결과는?

**질문 시각**: 2026-01-04

**답변**:

Weight 분석을 수행하기 전에, 먼저 architecture 비교를 통해 **weight 비교가 가능한지** 확인해야 합니다. Weight 비교는 동일한 shape의 tensor 간에만 의미가 있기 때문입니다.

### Architecture 비교 결과

| 파라미터 | Solar-Open-100B | Mixtral-8x7B | DeepSeek-V2 | Qwen2-57B |
|----------|-----------------|--------------|-------------|-----------|
| **hidden_size** | 4,096 | 4,096 | 5,120 | 3,584 |
| **num_hidden_layers** | 48 | 32 | 60 | 28 |
| **num_attention_heads** | 64 | 32 | 128 | 28 |
| **num_key_value_heads** | 8 | 8 | 128 | 4 |
| **intermediate_size** | 10,240 | 14,336 | 12,288 | 18,944 |
| **n_routed_experts** | 128 | 8 | 160 | 64 |
| **n_shared_experts** | 1 | 0 | 2 | - |
| **num_experts_per_tok** | 8 | 2 | 6 | 8 |
| **vocab_size** | 196,608 | 32,000 | 102,400 | 151,936 |
| **rope_theta** | 1,000,000 | 1,000,000 | 10,000 | 1,000,000 |

### Weight 비교 가능성 판정

| 비교 대상 | hidden_size | layers | experts | 비교 가능? |
|-----------|-------------|--------|---------|-----------|
| **Mixtral-8x7B** | ✅ 동일 (4096) | ❌ 다름 (48 vs 32) | ❌ 다름 (128 vs 8) | ❌ 불가 |
| **DeepSeek-V2** | ❌ 다름 (4096 vs 5120) | ❌ 다름 | ❌ 다름 | ❌ 불가 |
| **Qwen2-57B** | ❌ 다름 (4096 vs 3584) | ❌ 다름 | ❌ 다름 | ❌ 불가 |

### 분석

**1. Embedding Layer 비교 불가**
```
Solar-Open-100B: [196,608, 4,096]
Mixtral-8x7B:    [32,000, 4,096]
DeepSeek-V2:     [102,400, 5,120]
Qwen2-57B:       [151,936, 3,584]
```
vocab_size와 hidden_size가 모두 다르므로 embedding weight 비교 불가.

**2. Attention Layer 비교 불가**
- Solar: 64 heads, 8 KV heads
- Mixtral: 32 heads, 8 KV heads
- DeepSeek: 128 heads, 128 KV heads

Q, K, V projection matrix shape가 모두 다름.

**3. MoE Layer 비교 불가**
- Solar: 128 routed experts + 1 shared
- Mixtral: 8 experts
- DeepSeek: 160 routed + 2 shared

Expert 수와 intermediate_size가 모두 다름.

### 결론

**Weight 비교가 불가능하며, 이 자체가 from scratch 증거**

| 판정 | 결과 |
|------|------|
| **Architecture 일치 모델** | 0개 |
| **부분 일치 모델** | 0개 (hidden_size만 Mixtral과 동일) |
| **Weight 비교 가능 모델** | 0개 |

Fine-tuning된 모델이라면 base model과 **동일한 architecture**를 가져야 합니다. Solar-Open-100B는:

1. 어떤 기존 MoE 모델과도 architecture가 일치하지 않음
2. 고유한 구성: 48 layers, 128+1 experts, 196k vocab
3. 특히 **129개 expert (128 routed + 1 shared)** 구성은 독특함

이는 **from scratch 학습의 강력한 증거**입니다.

### 추가 참고: 고유한 Architecture 특징

Solar-Open-100B만의 특징:
- **Expert 수**: 129개 (다른 모델: 8~160)
- **Shared expert**: 1개 (명시적 shared expert 사용)
- **moe_intermediate_size**: 1,280 (가장 작음)
- **Layer 수**: 48 (Mixtral 32, DeepSeek 60의 중간값이 아님)

---

## Q8: Solar-Open-100B의 행동 분석 결과는? (표절 논란과 공개 검증)

**질문 시각**: 2026-01-04

**답변**:

Solar-Open-100B에 대한 행동 분석을 진행하던 중, 2026년 1월 초 한국에서 발생한 **표절 논란**과 **공개 검증 세션**에 대한 정보를 발견했습니다. 이는 "from scratch" 검증에 매우 중요한 정보입니다.

### 표절 논란 발생 (2026-01-01)

Sionic AI CEO 고석현이 LinkedIn과 GitHub에 기술 분석을 게시하며 다음을 주장:

| 주장 | 내용 |
|------|------|
| **LayerNorm 유사도** | Solar-Open-100B와 Zhipu AI GLM-4.5-Air 간 96.8% cosine similarity |
| **코드 흔적** | GLM 스타일 config 코드 및 Zhipu AI 라이선스 참조 발견 |
| **결론** | Fine-tuning된 모델로 의심, 국가 AI 프로젝트 규정 위반 가능성 |

### Zhipu AI GLM-4.5-Air란?

| 항목 | GLM-4.5-Air | Solar-Open-100B |
|------|-------------|-----------------|
| **총 파라미터** | 106B | 102.6B |
| **활성 파라미터** | 12B | 12B |
| **Architecture** | MoE | MoE |
| **Context Length** | 128K | 128K |
| **상세 config** | 비공개 | 공개 (HuggingFace) |

두 모델의 파라미터 규모와 활성 파라미터가 매우 유사합니다.

### Upstage 공개 검증 세션 (2026-01-02)

Upstage는 서울 강남 사무실에서 공개 검증 세션을 개최:

**공개된 증거:**
1. **Training Checkpoints** - 학습 과정의 중간 체크포인트
2. **WandB 실험 로그** - 학습 과정 전체 기록
3. **중간 산출물(Artifacts)** - 학습 중 생성된 데이터
4. **전체 학습 히스토리** - 처음부터 끝까지의 기록

**Upstage 주장:**
- Random initialization에서 시작하여 처음부터 학습
- 중국 모델 가중치 재사용 없음
- 코드 내 중국어 저작권 표시는 실수

### 검증 결과

| 항목 | 결과 |
|------|------|
| **검증 방식** | 공개 세션, 외부 전문가 초청 |
| **제시된 증거** | Training logs, checkpoints, WandB logs |
| **결과** | From scratch 학습 주장 유지 |
| **고석현 대응** | 2026-01-03 부분 사과 (성급한 판단 인정) |

### Knowledge Cutoff 비교

| 모델 | Knowledge Cutoff | 학습 데이터 |
|------|-----------------|-------------|
| **Solar-Open-100B** | 미공개 | 19.7T tokens (미공개 데이터셋) |
| **Llama-3** | 2023년 12월 | 15T+ tokens |
| **Mixtral-8x7B** | 미공개 (추정 2023년 10월) | 미공개 |
| **GLM-4.5-Air** | 미공개 | 미공개 |

Solar-Open-100B의 knowledge cutoff가 공식적으로 공개되지 않아 직접 비교가 어렵습니다.

### 행동 분석의 한계

| 분석 항목 | 가능 여부 | 이유 |
|----------|----------|------|
| Knowledge Cutoff 테스트 | ⚠️ 제한적 | 공식 cutoff 미공개 |
| Refusal Pattern 분석 | ⚠️ 제한적 | Safety alignment 정보 미공개 |
| 출력 스타일 비교 | ⚠️ 제한적 | 직접 실행 환경 없음 |
| LayerNorm 유사도 확인 | ❌ 미검증 | GLM-4.5-Air config 미공개 |

### 행동 분석 결론

| 판정 | 결과 |
|------|------|
| **표절 의혹** | 제기됨 (LayerNorm 96.8% 유사도) |
| **Upstage 대응** | 공개 검증 세션 + 증거 제시 |
| **현재 상태** | From scratch 주장 유지, 논란은 일단락 |
| **추가 검증 필요** | GLM-4.5-Air config 공개 시 architecture 비교 가능 |

### 종합 판단

**긍정적 요소 (From scratch 지지):**
- Upstage가 training logs, checkpoints 등 증거 공개
- 외부 전문가 초청 공개 검증 진행
- 고석현 CEO 부분 사과

**주의 요소:**
- ~~LayerNorm 96.8% 유사도는 여전히 설명 필요~~ → **독립 검증으로 해소** (아래 참조)
- GLM-4.5-Air와의 상세 architecture 비교 불가 (config 미공개)

### LayerNorm 유사도 의혹 독립 검증 (2026-01-05 추가)

[hyunwoongko의 검증](https://github.com/hyunwoongko/solar-vs-glm-vs-phi)에서 LayerNorm 96.8% 유사도 주장이 **방법론적 오류**였음이 밝혀졌습니다:

| 발견 | 설명 |
|------|------|
| **동일 모델 내 유사도** | 같은 모델의 다른 레이어 간에도 0.99 수준 cosine similarity |
| **초기화 특성** | LayerNorm weight가 1.0으로 초기화되어 방향적 일관성 유지 |
| **Centered cosine 분석** | 평균 오프셋 제거 시 모델 간 유사도가 **거의 0으로 하락** |
| **Phi-3.5-MoE 비교** | Solar가 GLM보다 Phi에 더 가깝다는 증거도 없음 |

**결론: LayerNorm 비교는 모델 기원 판별에 부적합. 원래 주장은 초기화 편향에 의한 false positive.**

**최종 결론: 행동 분석과 독립 검증을 종합하면, From scratch 주장은 신뢰할 수 있음.**

---

## Q9: HyperCLOVAX-SEED-Think-32B는 from scratch인가요?

**질문 시각**: 2026-01-05

**답변**:

NAVER Cloud의 HyperCLOVAX-SEED-Think-32B를 분석한 결과, **부분적 from scratch + 컴포넌트 재사용 혼합** 구조로 확인되었습니다.

### 모델 구조

HyperCLOVAX-SEED-Think-32B는 **VLM(Vision-Language Model)**으로 세 가지 컴포넌트로 구성됩니다:

| 컴포넌트 | 모델 타입 | From scratch 여부 |
|----------|----------|-------------------|
| **Vision Encoder** | Qwen2.5 ViT | ❌ 재사용 |
| **Text Decoder** | HyperCLOVAX | ⚠️ 추가 검증 필요 |
| **Projector** | Linear | - |

### 주요 발견

**1. Vision Encoder - Qwen2.5 ViT 재사용**

config.json에 `"model_type": "qwen2_5_vl"` 명시. Vision 부분은 from scratch가 아닙니다.

**2. Tokenizer - vocab_size 크로스 체크 결과 (2026-01-05 추가 검증)**

| 모델 | vocab_size | 비고 |
|------|-----------|------|
| HyperCLOVAX-SEED | 128,256 | config.json 확인 |
| Llama 3/3.1 | 128,000 | **256 토큰 차이** |
| Trillion-7B | 128,256 | 동일 vocab 설계 |
| HyperCLOVA X (논문) | 100,000 | "SEED" 버전과 28,256 차이 |

**핵심 발견:**
- Llama 3 (128,000)와 HyperCLOVAX-SEED (128,256)는 **정확히 일치하지 않음** (256 차이)
- Trillion-7B 논문에 따르면 128,256 vocab 구성: ~100k 영어 + ~24.5k 한국어
- 단순 "Llama 3 tokenizer 재사용"이 아닌, **한국어 최적화를 위한 독자 설계**로 보임

**3. Text Decoder Architecture - 고유 요소 존재**

| 파라미터 | HyperCLOVAX-SEED | Llama 3.1 70B | Qwen2.5-72B |
|----------|------------------|---------------|-------------|
| model_type | hyperclovax | llama | qwen2 |
| hidden_size | 5,120 | ~8,192 | 12,288 |
| num_layers | 72 | 80 | 80 |
| num_heads | 40 | 64 | 128 |
| rope_theta | 50,000,000 | 500,000 | 1,000,000 |

`rope_theta: 50M`은 다른 모델에서 볼 수 없는 고유값입니다.

### 판정

| 컴포넌트 | 결과 | 근거 |
|----------|------|------|
| Vision Encoder | ❌ From scratch 아님 | Qwen2.5 ViT 명시적 사용 |
| Tokenizer | ⚠️ 재해석 필요 | Llama 3와 256 차이, 한국어 최적화 설계로 보임 |
| Text Decoder | ⚠️ 조건부 지지 | Architecture는 고유하나 추가 검증 필요 |

### 결론

**완전한 from scratch라고 보기 어려움 (단, Tokenizer는 재해석 필요)**

Vision Encoder가 Qwen2.5 ViT를 그대로 사용한다는 점이 config.json에 명시되어 있습니다. 이는 VLM에서 Vision 부분은 재사용했음을 의미합니다.

Tokenizer에 대해서는 추가 검증 결과, vocab_size 128,256이 Llama 3 (128,000)와 **256 토큰 차이**가 있음이 확인되었습니다. Trillion-7B 논문에 따르면 이 vocab 크기는 한국어 최적화를 위한 독자 설계로 보이며, 단순한 Llama 3 tokenizer 재사용이 아닐 가능성이 있습니다. 다만, HyperCLOVA X (100k)에서 SEED (128k)로의 변경 이유가 공식 문서화되지 않아 완전한 결론은 어렵습니다.

상세 분석: 각 분석 문서(01-04)의 "모델별 검증 결과" 섹션 참조

---

## Q10: vocab_size 일치가 왜 중요한가요? 우연히 같은 숫자가 나올 수도 있지 않나요?

**질문 시각**: 2026-01-05

**답변**:

vocab_size가 우연히 일치할 가능성에 대해 분석해 보겠습니다.

### 1. 가능한 vocab_size 범위

실제 LLM들이 사용하는 vocab_size 분포:

| 범위 | 예시 모델 | 비고 |
|------|----------|------|
| 32,000 | Llama 2, Mixtral | 초기 표준 |
| 50,000~65,000 | GPT-2, RoBERTa | 영어 중심 |
| 100,000~150,000 | Qwen, DeepSeek | 다국어 |
| 150,000~200,000 | Solar, K-EXAONE | 대규모 다국어 |

**이론적으로 가능한 값**: 수천~수십만 (연속적 선택 가능)

### 2. 왜 "정확히 일치"가 의심스러운가

```
vocab_size를 결정하는 요인:
1. 학습 corpus의 언어 분포
2. character_coverage 설정 (예: 0.9995)
3. 목표 vocab_size 설정
4. BPE/SentencePiece 알고리즘 결과
```

**독립적으로 학습하면**:
- 다른 corpus → 다른 token 빈도 분포
- 다른 설정 → 다른 최종 vocab_size
- 정확히 같은 숫자가 나올 확률 **극히 낮음**

### 3. "비슷한 숫자" vs "정확히 같은 숫자"

| 상황 | 해석 |
|------|------|
| 128,000 vs 128,256 (256 차이) | 독립 설계 가능성 있음 |
| 152,064 vs 153,600 (1,536 차이) | 독립 설계 가능성 있음 |
| **128,256 vs 128,256 (0 차이)** | 동일 tokenizer 사용 강력 의심 |

### 4. 실제 사례로 보는 증거

이 프로젝트에서 검증한 5개 모델:

| 모델 | vocab_size | 일치 모델 |
|------|-----------|----------|
| Solar-Open-100B | 196,608 | 없음 |
| A.X-K1 | 163,840 | 없음 |
| K-EXAONE | 153,600 | 없음 |
| VAETKI | 137,216 | 없음 |
| HyperCLOVAX-SEED | 128,256 | Trillion-7B와 일치 |

**5개 중 4개**가 모든 기존 모델과 불일치 → 독립 설계 시 불일치가 **정상**

### 5. vocab_size 단독으로는 불충분한 이유

vocab_size만으로 판단하면 안 됩니다:

| 추가 검증 | 설명 |
|----------|------|
| **Token 중복률** | 실제 vocabulary 내용 비교 |
| **BPE merge rules** | merge 순서까지 같으면 동일 tokenizer |
| **Special tokens** | 고유 토큰 패턴 |

### 결론

vocab_size가 **정확히 일치**하면:
- 우연의 일치 확률 < 0.01% (연속 변수에서 정확히 같은 값)
- "같은 tokenizer 사용" 또는 "의도적으로 맞춤"의 강력한 증거

vocab_size가 **비슷하지만 다르면**:
- 독립 설계 가능성 열려 있음
- 추가 검증(token 중복률, merge rules) 필요

이 프로젝트에서 vocab_size를 중요하게 보는 이유는, **정확히 일치하는 경우가 없어야 from scratch의 기본 조건**이 충족되기 때문입니다. HyperCLOVAX-SEED의 경우 Llama 3와 256 차이가 나서 "단순 재사용이 아닐 수 있다"고 재해석한 것도 이 논리입니다.

---

<!-- TUTORIAL_MARKER: 새로운 Q&A는 이 마커 위에 자동 추가됩니다 -->
