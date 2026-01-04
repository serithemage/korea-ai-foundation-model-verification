# 튜토리얼: Q&A 형식 학습 기록

이 문서는 Solar-Open-100B 검증 과정에서 진행한 질문과 답변을 시간순으로 기록합니다.

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

<!-- TUTORIAL_MARKER: 새로운 Q&A는 이 마커 위에 자동 추가됩니다 -->
