# Architecture 분석

> 신뢰도: 중간 | 접근성: 높음 | Fine-tuning 탐지력: 양호

## 개요

Architecture 분석은 모델의 구조적 특징을 비교하여 기원을 추적합니다. 동일한 architecture config는 fine-tuning의 강력한 증거가 되며, 고유한 구조적 특징은 from scratch 학습을 시사합니다.

## 분석 항목

### 1. 기본 Hyperparameters 비교
- 레이어 수, hidden dimension, attention heads
- Intermediate size, vocabulary size

### 2. 활성화 함수 및 Normalization
- SiLU, GELU, ReLU 등
- RMSNorm vs LayerNorm

### 3. 고유한 구조적 특징
- RoPE scaling 방식
- Attention 구현 (GQA, MQA, MHA)
- MoE 구성 (expert 수, top-k)

## 고유성 판단 기준

### 동일 config 판정

다음 조건을 모두 만족하면 동일 architecture로 판정:

1. `hidden_size` 일치
2. `num_hidden_layers` 일치
3. `num_attention_heads` 일치
4. `intermediate_size` 일치
5. `hidden_act` 일치

### 파생 모델 가능성 지표

| 일치 항목 | 해석 |
|----------|------|
| 5/5 | 동일 architecture - fine-tuning 의심 |
| 3-4/5 | 유사 architecture - 참조 가능성 |
| 1-2/5 | 독립적 설계 가능성 |
| 0/5 | 완전히 다른 architecture |

---

## 모델별 검증 결과

### 1. Upstage Solar-Open-100B ✅

**검증일**: 2026-01-04

#### 기본 정보

| 항목 | 값 |
|------|-----|
| **모델 유형** | Mixture-of-Experts (MoE) |
| **총 파라미터** | 102.6B |
| **활성 파라미터** | 12B (토큰당) |
| **Expert 구성** | 129개 (128 routed + 1 shared, top-8 활성화) |
| **Context Length** | 128k tokens |

#### Architecture 비교 요약

| 파라미터 | Solar-Open-100B | Mixtral | DeepSeek-V2 | Qwen2-57B | 일치 모델 |
|----------|-----------------|---------|-------------|-----------|----------|
| hidden_size | 4,096 | 4,096 | 5,120 | 3,584 | Mixtral만 |
| num_layers | 48 | 32 | 60 | 28 | 없음 |
| num_heads | 64 | 32 | 128 | 28 | 없음 |
| num_kv_heads | 8 | 8 | 128 | 4 | Mixtral만 |
| n_experts | 128+1 | 8 | 160+2 | 64 | 없음 |
| vocab_size | 196,608 | 32,000 | 102,400 | 151,936 | 없음 |
| rope_theta | 1,000,000 | 1,000,000 | 10,000 | 1,000,000 | Mixtral, Qwen |

#### 판정

| 일치 항목 수 | 비교 대상 | 결과 |
|-------------|----------|------|
| **2/7** | Mixtral | hidden_size, kv_heads만 일치 |
| **1/7** | DeepSeek-V2 | rope_theta 계열만 유사 |
| **1/7** | Qwen2-57B | rope_theta만 동일 |

#### 고유 특징

1. **129개 Expert 구성** (128 routed + 1 shared) - 다른 모델에서 볼 수 없는 구성
2. **48 layers** - Mixtral(32)과 DeepSeek(60)의 단순 중간값 아님
3. **64 attention heads** - 가장 많은 head 수 (Dense 모델 제외)
4. **moe_intermediate_size: 1,280** - 비교 대상 중 가장 작음 (효율적 설계)
5. **vocab_size: 196,608** - 모든 비교 대상 중 가장 큼

**결론: 0/5 완전 일치 → 독립적 설계 (From scratch 지지)**

---

### 2. NAVER Cloud HyperCLOVAX-SEED-Think-32B ⚠️

**검증일**: 2026-01-05

#### 기본 정보

| 항목 | 값 |
|------|-----|
| **모델 유형** | Dense (Vision-Language Model) |
| **총 파라미터** | 32B (33B params) |
| **Context Length** | 128K tokens |
| **Knowledge Cutoff** | 2025년 5월 |

#### 컴포넌트 구조

HyperCLOVAX-SEED-Think-32B는 **VLM**으로 세 가지 컴포넌트로 구성됩니다:

```
┌─────────────────────────────────────────────────────────┐
│              HCXVisionV2ForCausalLM                     │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌──────────┐  ┌────────────────┐ │
│  │  Vision Encoder │→│ Projector│→│  Text Decoder   │ │
│  │  (Qwen2.5 ViT)  │  │ (Linear) │  │ (HyperCLOVAX)  │ │
│  └─────────────────┘  └──────────┘  └────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

#### Text Decoder Config 비교

| 파라미터 | HyperCLOVAX-SEED-32B | Llama 3.1 70B | Qwen2.5-72B |
|----------|---------------------|---------------|-------------|
| **model_type** | hyperclovax | llama | qwen2 |
| **hidden_size** | 5,120 | ~8,192 | 12,288 |
| **num_hidden_layers** | 72 | 80 | 80 |
| **num_attention_heads** | 40 | 64 | 128 |
| **num_key_value_heads** | 8 | 8 | 8 |
| **vocab_size** | 128,256 | 128,256 | ~152,000 |
| **rope_theta** | 50,000,000 | 500,000 | 1,000,000 |

#### Vision Encoder Config

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| **model_type** | qwen2_5_vl | **Qwen2.5 Vision Transformer 사용** |
| **hidden_size** | 1,280 | |
| **out_hidden_size** | 5,120 | Text decoder hidden_size와 일치 |
| **depth** | 32 | |
| **num_heads** | 16 | |

#### 고유 요소

1. `model_type: hyperclovax` - 고유한 모델 타입
2. `rope_theta: 50,000,000` - Llama 3 (500k), Qwen2.5 (1M)보다 훨씬 큼
3. `attention_multiplier: 0.08838834764831845` - 고유한 설정
4. 72 layers, 40 heads - 다른 모델과 일치하지 않는 조합

#### 판정

| 컴포넌트 | 결과 | From scratch 지지 |
|----------|------|------------------|
| **Text Decoder** | 고유 architecture | ✅ 지지 |
| **Vision Encoder** | Qwen2.5 ViT 사용 | ❌ 재사용 |
| **rope_theta** | 50M (고유값) | ✅ 지지 |
| **vocab_size** | Llama 3와 동일 | ⚠️ 의문점 |

**결론: 부분적 재사용 (Vision Encoder는 from scratch 아님)**

---

### 3. SKT A.X-K1 ✅

**검증일**: 2026-01-05

#### 기본 정보

| 항목 | 값 |
|------|-----|
| **모델 유형** | Mixture-of-Experts (MoE) |
| **model_type** | AXK1 (고유) |
| **총 파라미터** | 519B |
| **활성 파라미터** | ~22B (토큰당, top-8 experts) |
| **Expert 구성** | 193개 (192 routed + 1 shared, top-8 활성화) |
| **Context Length** | 131,072 tokens (YaRN RoPE scaling) |

#### Architecture 비교 요약

| 파라미터 | A.X-K1 | Solar-Open-100B | DeepSeek-V2 | Qwen2-57B | 일치 모델 |
|----------|--------|-----------------|-------------|-----------|----------|
| hidden_size | 7,168 | 4,096 | 5,120 | 3,584 | 없음 |
| num_layers | 61 | 48 | 60 | 28 | 없음 |
| num_heads | 64 | 64 | 128 | 28 | Solar만 |
| num_kv_heads | 64 (MHA) | 8 (GQA) | 128 | 4 | 없음 |
| n_experts | 192+1 | 128+1 | 160+2 | 64 | 없음 |
| experts_per_tok | 8 | 8 | 6 | 8 | Solar, Qwen |
| vocab_size | 163,840 | 196,608 | 102,400 | 151,936 | 없음 |
| rope_theta | 10,000 | 1,000,000 | 10,000 | 1,000,000 | DeepSeek만 |
| intermediate_size | 18,432 | N/A | 12,288 | 2,560 | 없음 |

#### Attention 구조

| 항목 | 값 | 비고 |
|------|-----|------|
| **Attention Type** | MHA (Multi-Head Attention) | num_heads = num_kv_heads = 64 |
| **Head Dimension** | 112 (7168 / 64) | |
| **Q Lora Rank** | 1,536 | Low-rank attention 사용 |
| **KV Lora Rank** | 512 | |

A.X-K1은 GQA가 아닌 **MHA(Multi-Head Attention)**을 사용하며, Low-rank projection을 적용합니다.

#### MoE 구조

| 항목 | 값 | 비고 |
|------|-----|------|
| **Routed Experts** | 192 | 가장 많은 expert 수 |
| **Shared Experts** | 1 | 모든 토큰에 활성화 |
| **Top-k** | 8 | Solar와 동일 |
| **MoE Intermediate Size** | 2,560 | |
| **Scoring Function** | softmax | |
| **Norm Top-k Prob** | True | |

#### RoPE Scaling (YaRN)

| 항목 | 값 |
|------|-----|
| **type** | yarn |
| **factor** | 4.0 |
| **original_max_position_embeddings** | 32,768 |
| **beta_fast** | 32.0 |
| **beta_slow** | 1.0 |
| **mscale** | 1.0 |
| **mscale_all_dim** | 0.0 |

YaRN scaling을 통해 32K → 131K context length 확장.

#### 고유 특징

1. **model_type: AXK1** - 완전히 고유한 모델 타입
2. **hidden_size: 7,168** - 모든 비교 대상 중 가장 큼
3. **193개 Expert 구성** (192 routed + 1 shared) - 가장 많은 expert 수
4. **MHA 사용** - 최신 MoE 모델들이 GQA를 선호하는 추세와 다름
5. **vocab_size: 163,840** - 모든 비교 대상과 불일치
6. **Low-rank Attention** - Q/KV에 LoRA rank 적용

#### 판정

| 일치 항목 수 | 비교 대상 | 결과 |
|-------------|----------|------|
| **1/9** | Solar-Open-100B | num_heads만 일치 |
| **1/9** | DeepSeek-V2 | rope_theta만 일치 |
| **1/9** | Qwen2-57B | experts_per_tok만 일치 |

**결론: 0/5 핵심 항목 완전 일치 → 독립적 설계 (From scratch 지지)**

---

### 4. NC AI VAETKI ✅

**검증일**: 2026-01-05

#### 기본 정보

| 항목 | 값 |
|------|-----|
| **모델 유형** | Mixture-of-Experts (MoE) |
| **model_type** | vaetki (고유) |
| **총 파라미터** | 112B |
| **활성 파라미터** | ~14B (토큰당, top-8 experts) |
| **Expert 구성** | 129개 (128 routed + 1 shared, top-8 활성화) |
| **Context Length** | 131,072 tokens (128K) |

#### Architecture 비교 요약

| 파라미터 | VAETKI | Solar-Open-100B | A.X-K1 | DeepSeek-V2 | 일치 모델 |
|----------|--------|-----------------|--------|-------------|----------|
| hidden_size | 3,072 | 4,096 | 7,168 | 5,120 | 없음 |
| num_layers | 48 | 48 | 61 | 60 | Solar만 |
| num_heads | 24 | 64 | 64 | 128 | 없음 |
| num_kv_heads | LoRA 방식 | 8 (GQA) | 64 (MHA) | 128 | 없음 |
| n_experts | 128+1 | 128+1 | 192+1 | 160+2 | Solar만 |
| experts_per_tok | 8 | 8 | 8 | 6 | Solar, A.X-K1 |
| vocab_size | 137,216 | 196,608 | 163,840 | 102,400 | 없음 |
| rope_theta | 10,000 | 1,000,000 | 10,000 | 10,000 | A.X-K1, DeepSeek |
| intermediate_size | 18,432 | N/A | 18,432 | 12,288 | A.X-K1만 |

#### Attention 구조 (고유)

| 항목 | 값 | 비고 |
|------|-----|------|
| **Attention Type** | Sliding + Full Hybrid | 고유한 하이브리드 방식 |
| **Sliding Window** | 512 tokens | |
| **Full Attention** | 매 6번째 layer (6, 12, 18, ...) | |
| **Head Dimension** | 64 (일반), 192 (QK) | |
| **Q Lora Rank** | 1,536 | Low-rank attention |
| **KV Lora Rank** | 512 | |
| **QK Rope Head Dim** | 64 | |
| **QK Nope Head Dim** | 128 | |

VAETKI는 **Sliding Window Attention과 Full Attention을 혼합**하는 고유한 구조를 사용합니다.

#### MoE 구조

| 항목 | 값 | 비고 |
|------|-----|------|
| **Routed Experts** | 128 | Solar와 동일 |
| **Shared Experts** | 1 | Solar와 동일 |
| **Top-k** | 8 | Solar, A.X-K1과 동일 |
| **MoE Intermediate Size** | 2,048 | |
| **Routed Scaling Factor** | 2.5 | |
| **First K Dense Replace** | 3 | 처음 3개 layer는 dense |
| **Norm Top-k Prob** | True | |

#### 고유 특징

1. **model_type: vaetki** - 완전히 고유한 모델 타입
2. **Hybrid Attention** - Sliding (512) + Full 혼합 (매 6번째 layer)
3. **작은 hidden_size** - 3,072로 비교 대상 중 가장 작음 (효율적 설계)
4. **vocab_size: 137,216** - 모든 비교 대상과 불일치
5. **고유한 Special Tokens** - `<|START|>`, `<|END|>`, `<|role_*|>` 스타일
6. **First K Dense** - 처음 3개 layer는 MoE 없이 dense로 처리

#### 판정

| 일치 항목 수 | 비교 대상 | 결과 |
|-------------|----------|------|
| **2/9** | Solar-Open-100B | num_layers, n_experts만 일치 |
| **2/9** | A.X-K1 | experts_per_tok, intermediate_size만 일치 |
| **1/9** | DeepSeek-V2 | rope_theta만 일치 |

**결론: 0/5 핵심 항목 완전 일치 → 독립적 설계 (From scratch 지지)**

---

### 5. LG AI 연구원 K-EXAONE ✅

**검증일**: 2026-01-05

#### 기본 정보

| 항목 | 값 |
|------|-----|
| **모델 유형** | Mixture-of-Experts (MoE) |
| **model_type** | exaone_moe (고유) |
| **총 파라미터** | 236B |
| **활성 파라미터** | ~23B (토큰당, top-8 experts) |
| **Expert 구성** | 129개 (128 routed + 1 shared, top-8 활성화) |
| **Context Length** | 262,144 tokens (256K) |

#### Architecture 비교 요약

| 파라미터 | K-EXAONE | Solar-Open-100B | A.X-K1 | VAETKI | 일치 모델 |
|----------|----------|-----------------|--------|--------|----------|
| hidden_size | 6,144 | 4,096 | 7,168 | 3,072 | 없음 |
| num_layers | 48 | 48 | 61 | 48 | Solar, VAETKI |
| num_heads | 64 | 64 | 64 | 24 | Solar, A.X-K1 |
| num_kv_heads | 8 (GQA) | 8 (GQA) | 64 (MHA) | LoRA | Solar만 |
| n_experts | 128+1 | 128+1 | 192+1 | 128+1 | Solar, VAETKI |
| experts_per_tok | 8 | 8 | 8 | 8 | 모두 동일 |
| vocab_size | 153,600 | 196,608 | 163,840 | 137,216 | 없음 |
| rope_theta | 1,000,000 | 1,000,000 | 10,000 | 10,000 | Solar만 |
| intermediate_size | 18,432 | N/A | 18,432 | 18,432 | A.X-K1, VAETKI |

#### Attention 구조

| 항목 | 값 | 비고 |
|------|-----|------|
| **Attention Type** | Sliding + Full Hybrid (LLLG 패턴) | 3 Sliding + 1 Full 반복 |
| **Sliding Window** | 128 tokens | |
| **GQA** | 8:1 (64 heads, 8 KV heads) | Solar와 동일 |
| **Head Dimension** | 128 | |
| **Scoring Function** | sigmoid | |

K-EXAONE은 **LLLG 패턴 (Local-Local-Local-Global)**으로 Sliding과 Full Attention을 혼합합니다.

#### MoE 구조

| 항목 | 값 | 비고 |
|------|-----|------|
| **Routed Experts** | 128 | Solar, VAETKI와 동일 |
| **Shared Experts** | 1 | Solar, VAETKI와 동일 |
| **Top-k** | 8 | 모든 국내 MoE와 동일 |
| **MoE Intermediate Size** | 2,048 | |
| **Routed Scaling Factor** | 2.5 | |
| **TopK Method** | noaux_tc | |
| **First/Last K Dense** | 1 | 첫 번째 layer만 dense |
| **is_moe_layer** | 첫 layer 제외 47개가 MoE | |

#### 고유 특징

1. **model_type: exaone_moe** - EXAONE 시리즈의 MoE 버전
2. **LLLG 패턴 Attention** - 3 Sliding + 1 Full 반복 (VAETKI와 다른 비율)
3. **262K Context** - 256K로 가장 긴 context length
4. **vocab_size: 153,600** - 모든 비교 대상과 불일치
5. **First Layer Dense** - 첫 번째 layer만 MoE 없이 dense
6. **GQA 8:1** - Solar와 동일한 효율적 KV 구조
7. **PII 토큰 내장** - `PI:EMAIL`, `PI:PHONE_NUM` 등 개인정보 마스킹 지원

#### 판정

| 일치 항목 수 | 비교 대상 | 결과 |
|-------------|----------|------|
| **4/9** | Solar-Open-100B | num_layers, num_heads, num_kv_heads, n_experts 일치 |
| **3/9** | VAETKI | num_layers, n_experts, intermediate_size 일치 |
| **2/9** | A.X-K1 | num_heads, intermediate_size 일치 |

**결론: 0/5 핵심 항목 완전 일치 → 독립적 설계 (From scratch 지지)**

> 참고: Solar와 일부 구조적 유사성(GQA 8:1, 129 experts)이 있으나, hidden_size, vocab_size가 다르고 Attention 패턴이 다르므로 독립 설계로 판정

---

## 참조용 모델 비교표

| 모델 | Type | Layers | Hidden | Heads | KV Heads | Experts | Vocab |
|------|------|--------|--------|-------|----------|---------|-------|
| **Solar-Open-100B** | MoE | 48 | 4,096 | 64 | 8 | 128+1 | 196,608 |
| **HyperCLOVAX-SEED** | Dense | 72 | 5,120 | 40 | 8 | - | 128,256 |
| **A.X-K1** | MoE | 61 | 7,168 | 64 | 64 | 192+1 | 163,840 |
| **VAETKI** | MoE | 48 | 3,072 | 24 | LoRA | 128+1 | 137,216 |
| **K-EXAONE** | MoE | 48 | 6,144 | 64 | 8 | 128+1 | 153,600 |
| Mixtral-8x7B | MoE | 32 | 4,096 | 32 | 8 | 8 | 32,000 |
| DeepSeek-V2 | MoE | 60 | 5,120 | 128 | 128 | 160+2 | 102,400 |
| Qwen2-57B-A14B | MoE | 28 | 3,584 | 28 | 4 | 64 | 151,936 |
| Llama-3-70B | Dense | 80 | 8,192 | 64 | 8 | - | 128,256 |

---

## 분석 코드

### config.json 비교

```python
from transformers import AutoConfig

def compare_configs(model_names):
    configs = {}
    for name in model_names:
        configs[name] = AutoConfig.from_pretrained(name)

    # 주요 항목 비교
    keys = [
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "max_position_embeddings",
        "rms_norm_eps",
        "rope_theta",
        "hidden_act",
    ]

    for key in keys:
        values = []
        for name in model_names:
            val = getattr(configs[name], key, "N/A")
            values.append(str(val))
        print(f"{key}: {values}")
```

### Attention 구조 분석

```python
def analyze_attention(config):
    num_heads = config.num_attention_heads
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)

    if num_kv_heads == num_heads:
        attn_type = "MHA (Multi-Head Attention)"
    elif num_kv_heads == 1:
        attn_type = "MQA (Multi-Query Attention)"
    else:
        attn_type = f"GQA (Grouped Query Attention, {num_heads//num_kv_heads} groups)"

    return attn_type
```

---

## 결론 도출 기준

**From scratch 지지 증거:**
- 모든 주요 모델과 architecture 불일치
- 고유한 MoE 구성 (129 experts)
- 비표준 hyperparameter 조합

**Fine-tuning 의심 증거:**
- 특정 모델과 완전한 config 일치
- 동일한 hidden_size, layers, heads 조합
- 표준적인 RoPE 설정
