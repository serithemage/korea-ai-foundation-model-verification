---
name: llm-dna-extraction-playbook
description: Use when planning, submitting, or debugging LLM-DNA fingerprint extraction jobs on SageMaker spot training. Trigger on keywords like "LLM-DNA jobs", "calc-dna", "submit_dna.py", "SageMaker spot", "Solar / K-EXAONE / A.X-K1 / EXAONE 추출", "Phase 5/6 재시도", "model fingerprint extraction", or follow-up verification rounds. Codifies 9 cost/operational lessons from Phase 4–6 of korea-ai-foundation-model-verification (2026-04-30, $64 / 77 jobs, 52.8% waste rate) so future rounds avoid the same learning curve.
---

# LLM-DNA Extraction Playbook

Operational checklist + entrypoint patches + yaml conventions for running LLM-DNA fingerprint extraction on SageMaker spot training without rediscovering the silent-failure traps that cost the first round ~$34.

**Authoritative source**: [docs/06-cost-and-architecture-lessons.md](../../docs/06-cost-and-architecture-lessons.md) — quantitative analysis (77 jobs, $64.28 total, $33.93 wasted).
**Project rule**: [CLAUDE.md "인프라 운영 규칙"](../../CLAUDE.md) section codifies the 9 rules below as *enforced project conventions* (every new model/job submission must pass them).
**Result narrative**: [docs/05-llm-dna-analysis.md](../../docs/05-llm-dna-analysis.md).
**Trap explanations** (for users): [docs/tutorial/05-방법론-평가.md Q17](../../docs/tutorial/05-방법론-평가.md), [docs/tutorial/01-기초개념.md Q15](../../docs/tutorial/01-기초개념.md).

## When to Use

- About to submit a SageMaker training job from `experiments/llm-dna/scripts/submit_dna.py`
- Adding a new model to `configs/models.yaml`
- Investigating a Failed/Stopped job in CloudWatch
- Planning Phase 2/3 (A.X-K1 519B, K-EXAONE-236B retry)
- Reviewing analyzer output before publishing distance/tree results

Skip if just reading existing tutorial Q&A or wiki pages — that work belongs to the [[llm-dna-knowledge]] skill.

## Pre-flight Checklist (run before every new model)

Each item maps to a real loss from Phase 4–6. Item count = lessons learned the hard way.

### 1. Sanity gate in entrypoint

`dna_train.py` must do a 1-prompt forward pass *between* model load and `calc-dna` invocation. If the test response is shorter than 20 chars, exit with non-zero code so SageMaker marks the job Failed (and spot retry kicks in automatically) rather than silently producing a Completed empty-response artifact.

```python
# After model.load(), before calc-dna subprocess:
test_out = wrapper.generate("Hello, can you respond?", max_new_tokens=50)
if len(test_out.strip()) < 20:
    log.error("Sanity gate failed: empty response, aborting before extraction")
    sys.exit(2)
```

This single gate catches: g7e×MoE `torch._grouped_mm` RuntimeError (CC < 9.0), base-model EOS-immediately bug, weight conversion failures (e.g. K-EXAONE-236B), HF auth misconfiguration. Without it, 8 K-EXAONE retries cost ~$13.

### 2. `max_new_tokens` cap is mandatory, not optional

llm-dna 0.2.x's `text_response_embeddings_random_projection_concat` extractor feeds responses to `all-mpnet-base-v2` (768-dim sentence-encoder), which truncates at ~512 tokens. Anything past that is wasted compute. Solar 1st run: 200s/prompt with cap=2048 (default); 2nd run: 24s/prompt with cap=256. **8× speedup, zero fingerprint loss.**

Apply via `dna_train.py:patch_model_wrapper_for_min_new_tokens(max_cap=256)` for all models (not just Solar). Also add to `configs/models.yaml` `dna_extraction` section as the default.

### 3. `--continue-on-error` requires response validation

Default `calc-dna --continue-on-error` lets jobs finish with 100/100 empty responses *and* report Status: Completed. CloudWatch shows the errors but external orchestration sees success. Add post-extraction validation in entrypoint:

```python
# After calc-dna, before S3 upload:
import json
with open(output_dir / "responses.json") as f:
    responses = json.load(f)["items"]
nonempty = sum(1 for r in responses if len(r.get("response", "").strip()) > 50)
if nonempty / len(responses) < 0.9:
    log.error("Sanity check failed: %d/%d nonempty (< 90%%)", nonempty, len(responses))
    sys.exit(3)
```

### 4. GPU compute capability matrix

`transformers >= 5.7` uses `torch._grouped_mm` for MoE forward — Hopper (CC 9.0) only. L40S (8.9), A100 (8.0), A10G (8.6) all RuntimeError on MoE models.

| Architecture | Min CC | Compatible spot instances |
|---|---|---|
| MoE (Mixtral, Solar-Open, MoE EXAONE, K-EXAONE-236B) | 9.0 | ml.p5.48xlarge, ml.p5e.48xlarge |
| Dense ≥ 70B fp16 | any | g7e.48xlarge (8×L40S), p5.48xlarge |
| Dense 7B–70B 4bit | any | g7e.4xl–12xl, p5.* |

Add `min_cuda_capability: 9.0` field to `configs/models.yaml` for MoE models, and have `submit_dna.py` reject mismatched instance choices before submission.

### 5. Multi-region quota — check, don't assume

`spot placement score = 9` means EC2 has spot capacity, but SageMaker has its own quota layer:

```bash
aws service-quotas list-service-quotas --service-code sagemaker --region <r> \
  --query "Quotas[?contains(QuotaName,'<inst>')].{Name:QuotaName,Value:Value}"
```

Phase 5 reality: us-east-1 `ml.p5.48xlarge for spot training job usage = 1` (serial only), us-west-2 = 0 (unusable). Two regions deployed via CDK (us-east-1 keeps short stack name; others get region suffix per `cdk/bin/llm-dna.ts`). `submit_dna.py --region` routes accordingly. Add automated quota check before submission so impossible jobs fail fast, not after upload.

### 6. Cold-start amortization

p5.48xl: ~9 min spot allocation + ECR pull, then 7–15 min weights load — all *non-billable* in spot. But each new job pays this fixed cost. For multiple small reference models, batch them in one container/instance instead of one job per model. Per-model amortized cost drops 60–70%.

```python
# Phase 2/3 batched job: load 3-4 dense reference models sequentially
# (Llama-8B + GLM-9B + Qwen-7B all fit in g7e.12xl 192GB fp16)
```

### 7. Quantization × device_map decisions are explicit

llm-dna 0.2.x auto-enables 8-bit for ≥ 7B models (`extraction.py:435-437`), then ModelWrapper pins to single GPU, then bnb refuses CPU offload — three layers of conflict. Always pass `--no-quantization` for fp16 paths and apply the `patch_model_wrapper_for_multi_gpu()` patch in `dna_train.py`. Use 4-bit only when explicit yaml `load_in: 4bit` and the model fits in single-device 4-bit on the smallest acceptable instance.

### 8. Result matching: latest-wins, always

S3 is append-only. Same `model_id` accumulates artifacts across job retries. `analyze.py`'s original `if model not in artifacts` matched *first* (oldest) — a silent bug that picked failed empty-response jobs over successful ones. Always:

```python
candidates: dict[str, list[str]] = {}
# ... collect all matches ...
artifacts = {model: max(keys) for model, keys in candidates.items()}  # latest by timestamp
```

Job prefixes contain `YYYYMMDD-HHMMSS`, so `max()` = chronological max. `.npy` should win over `.json` regardless of timestamp (richer format), but within the same format pick latest.

### 9. Outlier auto-detection in analyzer

If the distance matrix has a pair with cosine < 0.1, that's almost always a silent failure (fallback embedding collapse), not a real family. `analyze.py` should print a warning and refuse to publish until the human investigates. One line:

```python
suspicious = [(a, b, d) for a, b, d in pairs if d < 0.1]
if suspicious:
    print(f"⚠️  Suspiciously close pairs (< 0.1): {suspicious} — verify responses.json")
```

## Required Patches in `dna_train.py`

These three patches are the minimal set that makes a fresh llm-dna 0.2.x install behave correctly across the model lineup. Keep them as named functions so they're discoverable on subsequent reads.

| Function | Purpose | Without it |
|---|---|---|
| `patch_model_wrapper_for_multi_gpu()` | `device_map="auto"` for non-quantized models | 100B+ fp16 weights pin to cuda:0 → OOM |
| `patch_model_wrapper_for_min_new_tokens(max_cap=256)` | Force ≥ 50 tokens, cap at 256 | Base models emit empty response → silent failure; cap=2048 → 8× wall time |
| HF cache redirect to `/tmp` | Use 1.7TB instance store, not 120GB root | 100B+ models fail with "No space left" |

## Standard Job Submission Flow

1. **Yaml**: add model entry with `instance_type`, `load_in`, `min_cuda_capability` (if MoE), `trust_remote_code`
2. **Quota check**: `aws service-quotas list-service-quotas` for the chosen instance in target region
3. **Sanity dry-run** (recommended for new model architectures): `python scripts/submit_dna.py --model X --dry-run` to validate hyperparameters
4. **Submit**: `python scripts/submit_dna.py --region <r> --model X --instance ml.p5.48xlarge --max-wait-hours 8`
5. **First-prompt poll** (after ~5 min into Training): check CloudWatch for `s/prompt` metric; if > 60s and the model is < 100B, abort and investigate before letting it run 4 hours
6. **Post-completion**: download `responses.json`, verify `nonempty / total > 0.9` and `avg_len > 100` before considering the artifact valid

## When the Playbook Fails (escalation criteria)

If the same failure repeats deterministically (same error log, no environmental difference) — *stop after the second attempt*. Custom architectures with `trust_remote_code=True` and conversion errors are usually deterministic; reproducible failures are not flaky-spot-interrupt failures. Phase 5a's K-EXAONE retry pattern was the pre-playbook example: 8 attempts × ~$1.5 = $12 of redundant confirmation that transformers 5.x simply cannot load the checkpoint.

For deterministic failures, escalate to: (a) skip the model with documented reason, (b) substitute with a same-family alternate (LG K-EXAONE-236B → EXAONE-3.5-32B-Instruct), or (c) defer to a future round when upstream library compatibility may have changed.

## Cross-references

- [docs/06-cost-and-architecture-lessons.md](../../docs/06-cost-and-architecture-lessons.md) — quantitative analysis of where the $34 of waste came from
- [docs/05-llm-dna-analysis.md](../../docs/05-llm-dna-analysis.md) — Phase 4–6 narrative, results, judgments
- [docs/tutorial/05-방법론-평가.md Q17](../../docs/tutorial/05-방법론-평가.md) — six functional-fingerprint operational traps in user-facing form
- [docs/tutorial/01-기초개념.md Q14·Q15·Q16·Q18](../../docs/tutorial/01-기초개념.md) — algorithm depth + paper-vs-implementation divergence + 12-year-old explainer
- [.claude/skills/llm-dna-knowledge/SKILL.md](../llm-dna-knowledge/SKILL.md) — wiki-first routing convention for content questions
- `experiments/llm-dna/source/dna_train.py` — entrypoint with the three patches
- `experiments/llm-dna/scripts/submit_dna.py`, `analyze.py` — submission and analyzer with latest-wins fix
