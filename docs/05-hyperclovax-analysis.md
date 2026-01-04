# HyperCLOVAX-SEED-Think-32B 검증

> 검증 대상: NAVER Cloud HyperCLOVAX-SEED-Think-32B
> 검증 일자: 2026-01-05
> 상태: 검증 진행 중

## 모델 개요

| 항목 | 값 |
|------|-----|
| **모델 유형** | Dense (Vision-Language Model) |
| **총 파라미터** | 32B (33B params) |
| **Context Length** | 128K tokens |
| **Knowledge Cutoff** | 2025년 5월 |
| **HuggingFace** | [링크](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B) |

## 모델 구조

HyperCLOVAX-SEED-Think-32B는 **VLM(Vision-Language Model)**으로, 세 가지 컴포넌트로 구성됩니다:

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

---

## 1. Tokenizer 분석

### 주요 발견

| 항목 | HyperCLOVAX-SEED | HyperCLOVA X (논문) | Llama 3 |
|------|------------------|---------------------|---------|
| **vocab_size** | 128,256 | 100,000 | 128,256 |
| **tokenizer_class** | GPT2Tokenizer | morpheme-aware BPE | tiktoken BPE |
| **특수 토큰 수** | 256개 (128000-128255) | - | - |

### 분석

**주의 필요 사항:**
- HyperCLOVAX-SEED의 vocab_size(128,256)가 **Llama 3와 정확히 일치**
- NAVER의 HyperCLOVA X 논문에서는 vocab_size가 100,000으로 명시됨
- "SEED" 버전은 원래 HyperCLOVA X와 다른 tokenizer를 사용하는 것으로 보임

**특수 토큰 구성:**
- Vision/Multimedia: `<|IMAGE_PAD|>`, `<|VIDEO_PAD|>`
- Conversation: `<|im_start|>`, `<|im_end|>`, `<|stop|>`
- Code: `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`
- Jupyter: `<|jupyter_start|>`, `<|jupyter_code|>`, `<|jupyter_output|>`
- Tool: `<|tool_start|>`, `<|tool_call|>`, `<|tool_response|>`

### 판정

| 항목 | 결과 | From scratch 지지 |
|------|------|------------------|
| vocab_size | Llama 3와 동일 (128,256) | ⚠️ 의문점 |
| 특수 토큰 | 독자적 구성 | ✅ 지지 |

---

## 2. Architecture 분석

### Text Decoder (HyperCLOVAX) Config

| 파라미터 | HyperCLOVAX-SEED-32B | Llama 3.1 70B | Qwen2.5-72B |
|----------|---------------------|---------------|-------------|
| **model_type** | hyperclovax | llama | qwen2 |
| **hidden_size** | 5,120 | ~8,192 | 12,288 |
| **num_hidden_layers** | 72 | 80 | 80 |
| **num_attention_heads** | 40 | 64 | 128 |
| **num_key_value_heads** | 8 | 8 | 8 |
| **head_dim** | 128 | 128 | 128 |
| **intermediate_size** | 24,192 | ~14,336 | 49,152 |
| **vocab_size** | 128,256 | 128,256 | ~152,000 |
| **rope_theta** | 50,000,000 | 500,000 | 1,000,000 |
| **hidden_act** | silu | silu | silu |
| **rms_norm_eps** | 1e-05 | 1e-05 | 1e-06 |

### Vision Encoder Config

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| **model_type** | qwen2_5_vl | **Qwen2.5 Vision Transformer 사용** |
| **hidden_size** | 1,280 | |
| **out_hidden_size** | 5,120 | Text decoder hidden_size와 일치 |
| **depth** | 32 | |
| **num_heads** | 16 | |
| **patch_size** | 14 | |

### 분석

**독자적 요소:**
1. `model_type: hyperclovax` - 고유한 모델 타입
2. `rope_theta: 50,000,000` - Llama 3 (500k), Qwen2.5 (1M)보다 훨씬 큼
3. `attention_multiplier: 0.08838834764831845` - 고유한 설정
4. 72 layers, 40 heads - 다른 모델과 일치하지 않는 조합

**의문점:**
1. **Vision Encoder가 Qwen2.5 ViT를 그대로 사용** - config에 `qwen2_5_vl` 명시
2. vocab_size가 Llama 3와 정확히 동일

### 판정

| 항목 | 결과 | From scratch 지지 |
|------|------|------------------|
| Text Decoder | 고유한 architecture | ✅ 지지 |
| Vision Encoder | Qwen2.5 ViT 사용 | ❌ 재사용 |
| rope_theta | 50M (고유값) | ✅ 지지 |

---

## 3. 종합 판정

### From scratch 주장 검토

| 컴포넌트 | 판정 | 근거 |
|----------|------|------|
| **Text Decoder** | ⚠️ 조건부 지지 | Architecture는 고유하나 vocab_size가 Llama 3와 동일 |
| **Vision Encoder** | ❌ From scratch 아님 | Qwen2.5 ViT 명시적 사용 |
| **Tokenizer** | ⚠️ 추가 검증 필요 | vocab_size 128,256 = Llama 3 동일 |

### 주요 발견 사항

1. **Vision Encoder 재사용 확인**
   - config.json에 `"model_type": "qwen2_5_vl"` 명시
   - Qwen2.5 Vision Transformer를 그대로 사용
   - Vision 부분은 from scratch가 아님

2. **Tokenizer 유사성**
   - vocab_size 128,256은 Llama 3 계열과 정확히 동일
   - HyperCLOVA X 논문(100,000)과 불일치
   - tokenizer 재사용 가능성 있음

3. **Text Decoder 고유성**
   - `hyperclovax` 모델 타입
   - 72 layers, 40 heads 조합은 고유
   - rope_theta 50M은 다른 모델에서 보이지 않음

### 결론

**부분적 From scratch + 컴포넌트 재사용 혼합**

- Text Decoder(LLM 본체): From scratch 가능성 있음 (추가 검증 필요)
- Vision Encoder: **Qwen2.5 ViT 재사용 확인**
- Tokenizer: Llama 3 계열 재사용 가능성 (vocab_size 일치)

### 추가 검증 필요 사항

| 항목 | 방법 | 우선순위 |
|------|------|----------|
| Tokenizer vocabulary 직접 비교 | Llama 3 tokenizer와 token 단위 비교 | 높음 |
| Text Decoder weight 분석 | Llama 3.1과 cosine similarity 비교 | 중간 |
| Training 증거 확인 | NAVER 공식 발표, 논문 확인 | 높음 |

---

## 참고 자료

- [HuggingFace Model Card](https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-32B)
- [HyperCLOVA X Technical Report (arxiv)](https://arxiv.org/html/2404.01954v1)
- [HyperCLOVA X Think Technical Report](https://clova.ai/cdn/media/2025/06/HyperCLOVA_X_THINK_Technical_Report.pdf)
