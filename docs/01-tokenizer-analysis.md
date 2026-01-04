# Tokenizer 분석

> 신뢰도: 높음 | 접근성: 높음 | Fine-tuning 탐지력: 우수

## 개요

Tokenizer 분석은 LLM이 from scratch로 학습되었는지 판별하는 가장 접근성 높은 방법입니다. Fine-tuning 시 tokenizer를 재학습하는 경우가 거의 없기 때문에, tokenizer의 유사성은 모델 기원을 추적하는 강력한 지표가 됩니다.

## 분석 항목

### 1. Vocabulary 비교
- 기존 base model들과의 토큰 중복률 확인
- 고유 토큰 식별

### 2. BPE Merge Rules 분석
- Merge 순서 및 패턴 비교
- 동일한 merge rules는 같은 tokenizer 증거

### 3. 특수 토큰 패턴 비교
- `<eos>`, `<pad>`, `<bos>`, `<unk>` 등
- Chat template 토큰 (`<|im_start|>`, `[INST]` 등)

## Tokenizer 작동 원리

| 방식 | 특징 | 사용 모델 |
|------|------|----------|
| **BPE** (Byte Pair Encoding) | 빈도 기반으로 인접 문자쌍 병합 | GPT-2, RoBERTa |
| **WordPiece** | likelihood 최대화 기준 병합 | BERT |
| **SentencePiece** | 공백 포함 원시 텍스트 처리 (▁ 마커 사용) | T5, Gemma, Llama |

## Fine-tuning 시 Tokenizer를 재학습하지 않는 이유

1. **Embedding 호환성**: 새 vocabulary는 pre-trained embedding과 호환되지 않음
2. **비용**: Tokenizer 재학습은 전체 corpus 재처리 필요
3. **성능 저하 위험**: Vocabulary 변경 시 원래 도메인 성능 저하

## Vocabulary 중복률 해석 기준

| 중복률 | 해석 |
|--------|------|
| **>98%** | Fine-tuning 가능성 높음 (safety token 등 소량 추가만) |
| **90-98%** | Continued pre-training 또는 vocabulary 확장 |
| **<90%** | From scratch 학습 강력 증거 |

---

## 모델별 검증 결과

### 1. Upstage Solar-Open-100B ✅

**검증일**: 2026-01-04

#### Vocabulary 크기 비교

| 모델 | Vocab Size | Tokenizer Type | Solar 대비 |
|------|-----------|----------------|------------|
| **Solar-Open-100B** | **196,608** | SentencePiece (BPE) | - |
| Qwen2-72B | 152,064 | BPE | -23% |
| Llama-3 | 128,256 | tiktoken (BPE) | -35% |
| DeepSeek-V2 | 102,400 | BPE | -48% |
| Mixtral-8x7B | 32,000 | SentencePiece | -84% |

#### Special Tokens 비교

| 모델 | bos_token | eos_token | pad_token |
|------|-----------|-----------|-----------|
| **Solar-Open-100B** | `<s>` | `</s>` | `<pad>` |
| Llama-3 | `<\|begin_of_text\|>` | `<\|end_of_text\|>` | (없음) |
| Mixtral | `<s>` | `</s>` | (없음) |

#### 판정

| 지표 | 결과 | 해석 |
|------|------|------|
| Vocab Size 일치 | 0개 모델 | ✅ From scratch 지지 |
| Special Tokens | Mixtral과 유사 | ⚠️ 중립 |
| Tokenizer Type | 공통 방식 | ⚠️ 중립 |

**결론: From scratch 학습 주장 지지**

---

### 2. NAVER Cloud HyperCLOVAX-SEED-Think-32B ⚠️

**검증일**: 2026-01-05

#### Vocabulary 크기 비교

| 모델 | Vocab Size | 비고 |
|------|-----------|------|
| **HyperCLOVAX-SEED** | **128,256** | Llama 3와 동일 |
| Llama 3 | 128,256 | 정확히 일치 |
| HyperCLOVA X (논문) | 100,000 | "SEED" 버전과 다름 |

#### Special Tokens 비교

| 토큰 | 값 | 비고 |
|------|-----|------|
| `<\|IMAGE_PAD\|>` | Vision용 | VLM 특화 |
| `<\|im_start\|>`, `<\|im_end\|>` | Conversation | ChatML 스타일 |
| `<\|fim_prefix\|>`, `<\|fim_middle\|>`, `<\|fim_suffix\|>` | Code | Fill-in-the-middle |

#### 판정

| 지표 | 결과 | 해석 |
|------|------|------|
| **Vocab Size** | Llama 3와 동일 (128,256) | ⚠️ 의문점 |
| **Special Tokens** | 독자적 구성 | ✅ 지지 |
| **논문 불일치** | HyperCLOVA X(100k) vs SEED(128k) | ⚠️ 추가 검증 필요 |

**결론: 추가 검증 필요 (vocab_size가 Llama 3와 정확히 일치)**

---

### 3. SKT A.X-K1 📋

**검증 상태**: 대기 중

| 항목 | 값 |
|------|-----|
| **예상 Vocab Size** | 미확인 |
| **Tokenizer Type** | 미확인 |

---

### 4. NC AI VAETKI 📋

**검증 상태**: 대기 중

| 항목 | 값 |
|------|-----|
| **예상 Vocab Size** | 미확인 |
| **Tokenizer Type** | 미확인 |

---

### 5. LG AI 연구원 K-EXAONE 📋

**검증 상태**: 대기 중

| 항목 | 값 |
|------|-----|
| **예상 Vocab Size** | 미확인 |
| **Tokenizer Type** | 미확인 |

---

## 분석 코드

### 1. Vocabulary 비교

```python
from transformers import AutoTokenizer

base_tok = AutoTokenizer.from_pretrained("base-model")
target_tok = AutoTokenizer.from_pretrained("target-model")

base_vocab = set(base_tok.get_vocab().keys())
target_vocab = set(target_tok.get_vocab().keys())

overlap = len(base_vocab & target_vocab)
overlap_pct = (overlap / len(base_vocab)) * 100
print(f"중복률: {overlap_pct:.2f}%")

# 고유 토큰 확인
only_in_base = base_vocab - target_vocab
only_in_target = target_vocab - base_vocab
print(f"Base에만 있는 토큰: {len(only_in_base)}")
print(f"Target에만 있는 토큰: {len(only_in_target)}")
```

### 2. Merge Rules 비교 (BPE/SentencePiece)

```python
# merges가 동일하면 같은 tokenizer
base_merges = base_tok.backend_tokenizer.model.get_vocab()
target_merges = target_tok.backend_tokenizer.model.get_vocab()

# 첫 100개 merge 비교
merge_match = sum(1 for i in range(min(100, len(base_merges), len(target_merges)))
                  if list(base_merges.items())[i] == list(target_merges.items())[i])
print(f"첫 100개 merge 일치율: {merge_match}%")
```

### 3. Special Tokens 비교

```python
print("Base special tokens:", base_tok.special_tokens_map)
print("Target special tokens:", target_tok.special_tokens_map)

# 추가된 special tokens 확인
print("Added tokens:", target_tok.added_tokens_encoder)
```

---

## 결론 도출 기준

**From scratch 지지 증거:**
- 모든 주요 base model과 vocabulary 중복률 90% 미만
- 고유한 special token 체계
- 독자적인 merge rules

**Fine-tuning 의심 증거:**
- 특정 base model과 95% 이상 vocabulary 중복
- 동일한 special token 패턴
- merge rules 일치
