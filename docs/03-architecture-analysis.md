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

### 4. NC AI VAETKI 📋

**검증 상태**: 대기 중

| 항목 | 값 |
|------|-----|
| **모델 유형** | MoE |
| **총 파라미터** | 112B |
| **Architecture 분석** | 미수행 |

---

### 5. LG AI 연구원 K-EXAONE 📋

**검증 상태**: 대기 중

| 항목 | 값 |
|------|-----|
| **모델 유형** | MoE |
| **총 파라미터** | 236B |
| **Architecture 분석** | 미수행 |

---

## 참조용 모델 비교표

| 모델 | Type | Layers | Hidden | Heads | KV Heads | Experts | Vocab |
|------|------|--------|--------|-------|----------|---------|-------|
| **Solar-Open-100B** | MoE | 48 | 4,096 | 64 | 8 | 128+1 | 196,608 |
| **HyperCLOVAX-SEED** | Dense | 72 | 5,120 | 40 | 8 | - | 128,256 |
| **A.X-K1** | MoE | 61 | 7,168 | 64 | 64 | 192+1 | 163,840 |
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
