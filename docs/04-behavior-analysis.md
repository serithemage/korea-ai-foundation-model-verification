# 행동 분석

> 신뢰도: 중간 | 접근성: 높음 | Fine-tuning 탐지력: 보통

## 개요

행동 분석은 모델의 출력 패턴을 분석하여 학습 기원을 추론합니다. Fine-tuned 모델은 base model의 특성을 상속하는 경향이 있으며, 이를 통해 기원을 추적할 수 있습니다.

## 분석 항목

### 1. Knowledge Cutoff 테스트
- 특정 시점 이후 사건에 대한 지식 확인
- Base model과 동일한 cutoff는 fine-tuning 증거

### 2. Refusal Pattern 분석
- 거부 응답의 문구 및 패턴
- 특정 base model 특유의 refusal 스타일

### 3. Safety Alignment 특성
- 유해 콘텐츠 거부 방식
- 경계 케이스 처리 패턴

### 4. 출력 스타일 분석
- 응답 구조 및 형식
- 특정 표현이나 문구 사용 패턴

## 해석 기준

### Knowledge Cutoff

| 상황 | 해석 |
|------|------|
| Base model과 동일한 cutoff | Fine-tuning 의심 |
| Base model보다 최신 cutoff | Continued pre-training 또는 from scratch |
| 매우 최근 cutoff (2024 후반~) | From scratch 가능성 높음 |

### Refusal Pattern

| 상황 | 해석 |
|------|------|
| 특정 모델과 동일한 refusal 문구 | Fine-tuning 강력 의심 |
| 유사하지만 다른 문구 | 독립적 alignment 가능성 |
| 완전히 다른 스타일 | From scratch 증거 |

## 주의사항

1. **행동 분석의 한계**: Post-training으로 행동 수정 가능
2. **Alignment 오버라이드**: RLHF/DPO로 base 특성 변경 가능
3. **다국어 차이**: 언어별로 다른 패턴 나타날 수 있음

---

## 모델별 검증 결과

### 1. Upstage Solar-Open-100B ✅

**검증일**: 2026-01-04

#### 표절 논란 발생 (2026-01-01)

Sionic AI CEO 고석현이 Solar-Open-100B에 대한 기술 분석을 공개:

| 주장 | 내용 |
|------|------|
| **LayerNorm 유사도** | GLM-4.5-Air와 96.8% cosine similarity |
| **코드 흔적** | GLM 스타일 config 및 Zhipu AI 라이선스 참조 |
| **결론** | Fine-tuning 의심, 국가 프로젝트 규정 위반 가능성 |

#### 비교 대상: Zhipu AI GLM-4.5-Air

| 항목 | GLM-4.5-Air | Solar-Open-100B |
|------|-------------|-----------------|
| 총 파라미터 | 106B | 102.6B |
| 활성 파라미터 | 12B | 12B |
| Architecture | MoE | MoE |
| Context Length | 128K | 128K |
| 상세 config | **비공개** | 공개 |

#### Upstage 공개 검증 (2026-01-02)

서울 강남에서 공개 검증 세션 개최:

**제시된 증거:**
- Training checkpoints
- WandB 실험 로그
- 중간 산출물(Artifacts)
- 전체 학습 히스토리

**결과:**
- From scratch 학습 주장 유지
- 고석현 CEO 2026-01-03 부분 사과

#### LayerNorm 유사도 의혹 독립 검증 (2026-01-05)

[hyunwoongko의 검증](https://github.com/hyunwoongko/solar-vs-glm-vs-phi)에서 LayerNorm 96.8% 유사도 주장이 **방법론적 오류**였음이 밝혀졌습니다:

| 발견 | 설명 |
|------|------|
| **동일 모델 내 유사도** | 같은 모델의 다른 레이어 간에도 0.99 수준 cosine similarity |
| **초기화 특성** | LayerNorm weight가 1.0으로 초기화되어 방향적 일관성 유지 |
| **Centered cosine 분석** | 평균 오프셋 제거 시 모델 간 유사도가 **거의 0으로 하락** |
| **Phi-3.5-MoE 비교** | Solar가 GLM보다 Phi에 더 가깝다는 증거도 없음 |

**결론**: Cosine similarity만으로는 LayerNorm 비교가 신뢰할 수 없음. 원래 주장은 초기화 편향에 의한 **false positive**.

#### 판정

| 요소 | From scratch 지지 | 주의 필요 |
|------|------------------|----------|
| 공개 검증 | ✅ Training logs 제시 | - |
| 외부 검증 | ✅ 전문가 초청 | - |
| 표절 의혹 대응 | ✅ 고석현 부분 사과 | - |
| LayerNorm 유사도 | ✅ 독립 검증으로 해소 | - |
| GLM 비교 | - | Config 미공개로 비교 불가 |

**결론: 행동 분석과 독립 검증을 종합하면, From scratch 주장은 신뢰할 수 있음**

---

### 2. NAVER Cloud HyperCLOVAX-SEED-Think-32B ⚠️

**검증일**: 2026-01-05

#### Knowledge Cutoff 정보

| 항목 | 값 |
|------|-----|
| **Knowledge Cutoff** | 2025년 5월 (공식 발표) |
| **공개일** | 2025년 6월 |

Knowledge cutoff가 2025년 5월로 공개되어 있어, 최신 데이터로 학습되었음이 확인됩니다.

#### 행동 분석 한계

| 분석 항목 | 상태 | 이유 |
|----------|------|------|
| Knowledge Cutoff 테스트 | ✅ 가능 | 2025년 5월 공개 |
| Refusal Pattern 분석 | ⚠️ 미수행 | 직접 실행 환경 없음 |
| 출력 스타일 비교 | ⚠️ 미수행 | 직접 실행 환경 없음 |

#### 판정

| 요소 | 결과 | 해석 |
|------|------|------|
| Knowledge Cutoff | 2025년 5월 | ✅ 최신 (독립 학습 가능성) |
| Vision Encoder | Qwen2.5 ViT 사용 | ❌ 컴포넌트 재사용 |
| Text Decoder | 행동 테스트 미수행 | ⚠️ 추가 검증 필요 |

**결론: Knowledge cutoff는 최신이나, Vision Encoder 재사용이 확인됨**

---

### 3. SKT A.X-K1 ✅

**검증일**: 2026-01-05

#### 행동 분석 한계

| 분석 항목 | 상태 | 이유 |
|----------|------|------|
| Knowledge Cutoff 테스트 | ⚠️ 미수행 | 직접 실행 환경 없음 |
| Refusal Pattern 분석 | ⚠️ 미수행 | 직접 실행 환경 없음 |
| 출력 스타일 비교 | ⚠️ 미수행 | 직접 실행 환경 없음 |

#### 특수 토큰에서 추론 가능한 정보

| 토큰 | 의미 |
|------|------|
| `<\|think\|>`, `</think>` | Chain-of-thought reasoning 지원 |
| `<\|image\|>`, `<\|video_*\|>` | Multimodal (VLM) 준비 |
| ChatML 스타일 (`<\|im_start\|>`, `<\|im_end\|>`) | 대화형 인터페이스 |

#### 판정

| 요소 | 결과 | 해석 |
|------|------|------|
| Tokenizer | vocab_size 163,840 (고유) | ✅ From scratch 지지 |
| Architecture | 완전히 고유한 구성 | ✅ From scratch 지지 |
| 행동 테스트 | 미수행 | ⚠️ 추가 검증 가능 |

**결론: Architecture와 Tokenizer 분석만으로 From scratch 신뢰 가능**

---

### 4. NC AI VAETKI ✅

**검증일**: 2026-01-05

#### 행동 분석 한계

| 분석 항목 | 상태 | 이유 |
|----------|------|------|
| Knowledge Cutoff 테스트 | ⚠️ 미수행 | 직접 실행 환경 없음 |
| Refusal Pattern 분석 | ⚠️ 미수행 | 직접 실행 환경 없음 |
| 출력 스타일 비교 | ⚠️ 미수행 | 직접 실행 환경 없음 |

#### 특수 토큰에서 추론 가능한 정보

| 토큰 | 의미 |
|------|------|
| `<think>`, `</think>` | Chain-of-thought reasoning 지원 |
| `<tool_start>`, `<tool_end>` | Tool/Function calling 지원 |
| `<\|role_start\|>`, `<\|role_end\|>` | 대화 역할 구분 |
| `<\|START\|>`, `<\|END\|>` | 고유한 BOS/EOS 스타일 |

#### 판정

| 요소 | 결과 | 해석 |
|------|------|------|
| Tokenizer | vocab_size 137,216 (고유) | ✅ From scratch 지지 |
| Architecture | 완전히 고유한 구성 (Hybrid Attention) | ✅ From scratch 지지 |
| 행동 테스트 | 미수행 | ⚠️ 추가 검증 가능 |

**결론: Architecture와 Tokenizer 분석만으로 From scratch 신뢰 가능**

---

### 5. LG AI 연구원 K-EXAONE ✅

**검증일**: 2026-01-05

#### 행동 분석 한계

| 분석 항목 | 상태 | 이유 |
|----------|------|------|
| Knowledge Cutoff 테스트 | ⚠️ 미수행 | 직접 실행 환경 없음 |
| Refusal Pattern 분석 | ⚠️ 미수행 | 직접 실행 환경 없음 |
| 출력 스타일 비교 | ⚠️ 미수행 | 직접 실행 환경 없음 |

#### 특수 토큰에서 추론 가능한 정보

| 토큰 | 의미 |
|------|------|
| `[BOS]`, `[PAD]` | 브라켓 스타일 (EXAONE 고유) |
| `<\|endofturn\|>` | 턴 종료 토큰 (고유 스타일) |
| `<\|system\|>`, `<\|user\|>`, `<\|assistant\|>` | 대화 역할 구분 |
| `<think>`, `</think>` | Chain-of-thought reasoning 지원 |
| `<tool_call>` | Tool/Function calling 지원 |
| `<vision>`, `<\|image_pad\|>`, `<\|video_pad\|>` | Multimodal (VLM) 지원 |
| `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>` | Fill-in-the-middle (Code) 지원 |

#### 판정

| 요소 | 결과 | 해석 |
|------|------|------|
| Tokenizer | vocab_size 153,600 (고유) | ✅ From scratch 지지 |
| Architecture | EXAONE 고유 구성 (LLLG Attention) | ✅ From scratch 지지 |
| Special Tokens | EXAONE 고유 스타일 (`[BOS]`, `<\|endofturn\|>`) | ✅ From scratch 지지 |
| 행동 테스트 | 미수행 | ⚠️ 추가 검증 가능 |

**결론: Architecture와 Tokenizer 분석만으로 From scratch 신뢰 가능**

---

## Knowledge Cutoff 비교표

| 모델 | Knowledge Cutoff | Training Data | 상태 |
|------|-----------------|---------------|------|
| **Solar-Open-100B** | 미공개 | 19.7T tokens | ✅ 완료 |
| **HyperCLOVAX-SEED** | 2025년 5월 | 미공개 | ⚠️ 진행중 |
| **A.X-K1** | 미확인 | 미확인 | ✅ 완료 |
| **VAETKI** | 미확인 | 미확인 | ✅ 완료 |
| **K-EXAONE** | 미확인 | 미확인 | ✅ 완료 |
| Llama-3 | 2023년 12월 | 15T+ tokens | (참조) |
| Mixtral-8x7B | 미공개 | 미공개 | (참조) |

---

## 분석 코드

### Knowledge Cutoff 테스트

```python
knowledge_test_events = [
    {"date": "2023-03", "event": "GPT-4 출시", "question": "GPT-4는 언제 출시되었나요?"},
    {"date": "2023-07", "event": "Llama 2 공개", "question": "Meta의 Llama 2 모델에 대해 알고 있나요?"},
    {"date": "2024-04", "event": "Llama 3 공개", "question": "Meta Llama 3에 대해 알고 있나요?"},
    {"date": "2025-01", "event": "최신 이벤트", "question": "2025년 1월에 발표된 AI 모델은?"},
]

def test_knowledge_cutoff(model, tokenizer, events):
    results = []
    for event in events:
        prompt = f"질문: {event['question']}\n답변:"
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({"date": event['date'], "knows": analyze_response(response, event['event'])})
    return results
```

### Refusal Pattern 분석

```python
known_refusal_patterns = {
    "llama": ["I cannot provide", "I'm not able to", "I can't assist with"],
    "claude": ["I don't feel comfortable", "I'd prefer not to", "I can't help with"],
    "gpt": ["I'm sorry, but I can't", "I'm not able to assist", "I cannot help with"],
}

def analyze_refusal_pattern(response):
    response_lower = response.lower()
    for model_type, patterns in known_refusal_patterns.items():
        for pattern in patterns:
            if pattern.lower() in response_lower:
                return model_type, pattern
    return "unknown", None
```

---

## 결론 도출 기준

**From scratch 지지 증거:**
- 모든 base model과 다른 knowledge cutoff
- 고유한 refusal 패턴
- 독자적인 응답 스타일

**Fine-tuning 의심 증거:**
- 특정 model과 동일한 knowledge cutoff
- 일치하는 refusal 문구
- 유사한 출력 구조
