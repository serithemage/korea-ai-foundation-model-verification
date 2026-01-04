# Weight 분석

> 신뢰도: 높음 | 접근성: 중간 | Fine-tuning 탐지력: 양호

## 개요

Weight 분석은 모델의 가중치를 직접 비교하여 from scratch 학습 여부를 판별합니다. Fine-tuned 모델은 base model과 높은 가중치 유사성을 보이는 반면, from scratch 모델은 독립적인 가중치 분포를 갖습니다.

## 분석 항목

### 1. Layer별 Cosine Similarity 계산
- 각 layer의 weight tensor 간 유사도 측정
- Fine-tuned 모델: 초기 레이어 90% 이상 유사도

### 2. Weight Tensor 해시 비교
- SHA-256 해시로 동일성 확인
- 완전히 동일한 layer 탐지

### 3. PCA를 통한 Weight 분포 분석
- Weight matrix의 주성분 분석
- From scratch: orthogonal 분포
- Fine-tuned: base model 근처 clustering

## 해석 기준

### Cosine Similarity 기준

| 평균 유사도 | 해석 |
|-------------|------|
| **>0.95** | 거의 확실히 fine-tuning |
| **0.8-0.95** | Fine-tuning 또는 continued pre-training |
| **0.5-0.8** | 부분적 weight 재사용 가능성 |
| **<0.5** | From scratch 가능성 높음 |

### Layer별 패턴

| 패턴 | 의미 |
|------|------|
| 초기 layer 높은 유사도, 후기 layer 낮음 | 전형적인 fine-tuning |
| 전체적으로 낮은 유사도 | From scratch 증거 |
| 일부 layer만 높은 유사도 | 부분적 weight 초기화 |

---

## 모델별 검증 결과

### 1. Upstage Solar-Open-100B ✅

**검증일**: 2026-01-04

#### Architecture 비교를 통한 Weight 비교 가능성 분석

Weight 비교는 동일한 shape의 tensor 간에만 의미가 있습니다.

| 파라미터 | Solar-Open-100B | Mixtral-8x7B | DeepSeek-V2 | Qwen2-57B |
|----------|-----------------|--------------|-------------|-----------|
| **hidden_size** | 4,096 | 4,096 | 5,120 | 3,584 |
| **num_hidden_layers** | 48 | 32 | 60 | 28 |
| **num_attention_heads** | 64 | 32 | 128 | 28 |
| **n_routed_experts** | 128 | 8 | 160 | 64 |
| **vocab_size** | 196,608 | 32,000 | 102,400 | 151,936 |

#### 판정

| 비교 대상 | Weight 비교 가능? | 이유 |
|-----------|------------------|------|
| Mixtral-8x7B | ❌ 불가 | layers, heads, experts 모두 다름 |
| DeepSeek-V2 | ❌ 불가 | hidden_size부터 다름 |
| Qwen2-57B | ❌ 불가 | 모든 dimension 다름 |

#### 결론

**Weight 비교 불가 → From scratch 증거**

Fine-tuning된 모델이라면 base model과 동일한 architecture를 가져야 합니다.
Solar-Open-100B는 어떤 기존 모델과도 architecture가 일치하지 않으므로,
**직접적인 weight 비교 없이도 from scratch 학습임을 강력히 시사**합니다.

---

### 2. NAVER Cloud HyperCLOVAX-SEED-Think-32B ⚠️

**검증일**: 2026-01-05

#### 컴포넌트별 Weight 비교 가능성

HyperCLOVAX-SEED는 VLM으로, 세 가지 컴포넌트로 구성됩니다:

| 컴포넌트 | 비교 대상 | Weight 비교 가능? |
|----------|----------|------------------|
| **Vision Encoder** | Qwen2.5 ViT | ✅ 가능 (동일 model_type 명시) |
| **Text Decoder** | Llama 3.1 70B | ⚠️ 부분 가능 (hidden_size 다름) |
| **Projector** | - | 새로 학습된 부분 |

#### Text Decoder Architecture 비교

| 파라미터 | HyperCLOVAX-SEED | Llama 3.1 70B | Qwen2.5-72B |
|----------|------------------|---------------|-------------|
| **hidden_size** | 5,120 | ~8,192 | 12,288 |
| **num_layers** | 72 | 80 | 80 |
| **num_heads** | 40 | 64 | 128 |
| **vocab_size** | 128,256 | 128,256 | ~152,000 |

#### 판정

| 컴포넌트 | 결과 | 해석 |
|----------|------|------|
| Vision Encoder | Qwen2.5 ViT 재사용 명시 | ❌ From scratch 아님 |
| Text Decoder | Architecture 불일치 | ⚠️ 추가 검증 필요 |
| Tokenizer | vocab_size 일치 (Llama 3) | ⚠️ 의문점 |

**결론: Vision Encoder는 재사용 확인, Text Decoder는 추가 검증 필요**

---

### 3. SKT A.X-K1 ✅

**검증일**: 2026-01-05

#### Architecture 비교를 통한 Weight 비교 가능성 분석

| 파라미터 | A.X-K1 | Solar-Open-100B | DeepSeek-V2 | Qwen2-57B |
|----------|--------|-----------------|-------------|-----------|
| **hidden_size** | 7,168 | 4,096 | 5,120 | 3,584 |
| **num_hidden_layers** | 61 | 48 | 60 | 28 |
| **num_attention_heads** | 64 | 64 | 128 | 28 |
| **num_kv_heads** | 64 | 8 | 128 | 4 |
| **n_routed_experts** | 192 | 128 | 160 | 64 |
| **vocab_size** | 163,840 | 196,608 | 102,400 | 151,936 |

#### 판정

| 비교 대상 | Weight 비교 가능? | 이유 |
|-----------|------------------|------|
| Solar-Open-100B | ❌ 불가 | hidden_size, layers, experts 모두 다름 |
| DeepSeek-V2 | ❌ 불가 | hidden_size, num_layers 다름 |
| Qwen2-57B | ❌ 불가 | 모든 dimension 다름 |

#### 결론

**Weight 비교 불가 → From scratch 증거**

A.X-K1은 **hidden_size 7,168**로 비교 대상 모델 중 가장 크며, **192개 routed experts**로 가장 많은 expert 수를 가집니다. 또한 GQA가 아닌 **MHA(num_kv_heads = num_heads = 64)**를 사용하여 아키텍처 구조가 완전히 다릅니다.

어떤 기존 모델과도 architecture가 일치하지 않으므로, **직접적인 weight 비교 없이도 from scratch 학습임을 강력히 시사**합니다.

---

### 4. NC AI VAETKI ✅

**검증일**: 2026-01-05

#### Architecture 비교를 통한 Weight 비교 가능성 분석

| 파라미터 | VAETKI | Solar-Open-100B | A.X-K1 | DeepSeek-V2 |
|----------|--------|-----------------|--------|-------------|
| **hidden_size** | 3,072 | 4,096 | 7,168 | 5,120 |
| **num_hidden_layers** | 48 | 48 | 61 | 60 |
| **num_attention_heads** | 24 | 64 | 64 | 128 |
| **n_routed_experts** | 128 | 128 | 192 | 160 |
| **vocab_size** | 137,216 | 196,608 | 163,840 | 102,400 |

#### 판정

| 비교 대상 | Weight 비교 가능? | 이유 |
|-----------|------------------|------|
| Solar-Open-100B | ❌ 불가 | hidden_size, heads 다름 |
| A.X-K1 | ❌ 불가 | hidden_size, layers, experts 모두 다름 |
| DeepSeek-V2 | ❌ 불가 | 모든 dimension 다름 |

#### 결론

**Weight 비교 불가 → From scratch 증거**

VAETKI는 **hidden_size 3,072**로 비교 대상 중 가장 작으며, **Sliding + Full Hybrid Attention**이라는 고유한 attention 방식을 사용합니다. 또한 **처음 3개 layer를 dense로 처리**하는 독특한 설계를 가지고 있습니다.

어떤 기존 모델과도 architecture가 일치하지 않으므로, **직접적인 weight 비교 없이도 from scratch 학습임을 강력히 시사**합니다.

---

### 5. LG AI 연구원 K-EXAONE ✅

**검증일**: 2026-01-05

#### Architecture 비교를 통한 Weight 비교 가능성 분석

| 파라미터 | K-EXAONE | Solar-Open-100B | A.X-K1 | VAETKI | DeepSeek-V2 |
|----------|----------|-----------------|--------|--------|-------------|
| **hidden_size** | 6,144 | 4,096 | 7,168 | 3,072 | 5,120 |
| **num_hidden_layers** | 48 | 48 | 61 | 48 | 60 |
| **num_attention_heads** | 64 | 64 | 64 | 24 | 128 |
| **num_kv_heads** | 8 | 8 | 64 | 8 | 128 |
| **n_routed_experts** | 128 | 128 | 192 | 128 | 160 |
| **vocab_size** | 153,600 | 196,608 | 163,840 | 137,216 | 102,400 |

#### 판정

| 비교 대상 | Weight 비교 가능? | 이유 |
|-----------|------------------|------|
| Solar-Open-100B | ❌ 불가 | hidden_size 다름 (6,144 vs 4,096) |
| A.X-K1 | ❌ 불가 | hidden_size, layers, num_kv_heads 다름 |
| VAETKI | ❌ 불가 | hidden_size, heads 다름 |
| DeepSeek-V2 | ❌ 불가 | hidden_size, layers 다름 |

#### 결론

**Weight 비교 불가 → From scratch 증거**

K-EXAONE은 **hidden_size 6,144**로 고유한 크기를 가지며, **LLLG 패턴 (3 Sliding + 1 Full Attention)**이라는 EXAONE 고유의 attention 방식을 사용합니다. **262K context length**는 비교 대상 중 가장 깁니다.

어떤 기존 모델과도 architecture가 일치하지 않으므로, **직접적인 weight 비교 없이도 from scratch 학습임을 강력히 시사**합니다.

---

## 분석 코드

### 1. Cosine Similarity 계산

```python
import torch
from transformers import AutoModel
import torch.nn.functional as F

def load_model_weights(model_name):
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)
    return {name: param.data for name, param in model.named_parameters()}

base_weights = load_model_weights("base-model")
target_weights = load_model_weights("target-model")

def cosine_sim(w1, w2):
    w1_flat = w1.flatten().float()
    w2_flat = w2.flatten().float()
    return F.cosine_similarity(w1_flat.unsqueeze(0), w2_flat.unsqueeze(0)).item()

# Layer별 유사도 계산
similarities = {}
for name in base_weights:
    if name in target_weights:
        if base_weights[name].shape == target_weights[name].shape:
            sim = cosine_sim(base_weights[name], target_weights[name])
            similarities[name] = sim
            print(f"{name}: {sim:.4f}")

# 평균 유사도
avg_sim = sum(similarities.values()) / len(similarities)
print(f"\n평균 유사도: {avg_sim:.4f}")
```

### 2. Weight Tensor 해시 비교

```python
import hashlib
import numpy as np

def weight_hash(tensor):
    """Weight tensor의 SHA-256 해시 계산"""
    arr = tensor.cpu().numpy().tobytes()
    return hashlib.sha256(arr).hexdigest()[:16]

# 동일한 weight 탐지
identical_layers = []
for name in base_weights:
    if name in target_weights:
        if base_weights[name].shape == target_weights[name].shape:
            base_hash = weight_hash(base_weights[name])
            target_hash = weight_hash(target_weights[name])
            if base_hash == target_hash:
                identical_layers.append(name)
                print(f"동일: {name}")

print(f"\n동일한 layer 수: {len(identical_layers)}")
```

---

## 주의사항

1. **메모리 요구사항**: 100B 모델 비교는 상당한 GPU/CPU 메모리 필요
2. **MoE 구조 고려**: Expert weight는 별도 분석 필요
3. **Quantization 영향**: 양자화된 모델은 해시 비교 불가

---

## 결론 도출 기준

**From scratch 지지 증거:**
- 모든 base model과 평균 cosine similarity 0.5 미만
- 동일한 layer 없음 (해시 불일치)
- PCA에서 독립적인 분포

**Fine-tuning 의심 증거:**
- 특정 base model과 0.9 이상 유사도
- 다수의 동일 layer (해시 일치)
- 초기 layer 높은 유사도 패턴
