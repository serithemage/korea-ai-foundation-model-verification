---
name: llm-dna-knowledge
description: Use when working on LLM model lineage, provenance, phylogenetic analysis, from-scratch verification, the project's verification targets (Solar-Open-100B, K-EXAONE, A.X-K1, HyperCLOVAX), reference families (Llama 3, Qwen 2.5, GLM, Mixtral, DeepSeek), or anything under experiments/llm-dna/. Routes through the project wiki/ before external search.
---

# llm-dna-knowledge

Project-local convention: always consult the curated `wiki/` knowledge base (28 cross-linked pages) before answering or acting on LLM lineage and verification topics. The wiki is compiled from primary sources; rebuilding context from scratch each session wastes tokens and produces contradictions.

## When to Use

- Questions about LLM-DNA, neighbor-joining, UPGMA, distance metrics, functional fingerprint
- Verification methodology: tokenizer / weight / architecture / behavior analysis
- The 3 dokpamo targets: Solar-Open-100B, K-EXAONE-236B, A.X-K1
- Reference clusters: Llama, Qwen, GLM, Mixtral, DeepSeek (in lineage context)
- Edits or scripts under `experiments/llm-dna/`
- "from-scratch" controversy, Phase 1/2 evaluations, Stanford AI Index entries

Skip when the project has no `wiki/` directory — fall back to external sources only.

## Workflow

1. **Read `wiki/index.md` first.** It groups all 28 pages by `type` (concept / entity / methodology / infra / policy). Match the user's keywords to 3–5 candidate pages.
2. **2-hop expansion via `[[wiki-link]]`.** Open each candidate, follow inline links one or two levels deep until you have full context. Example: "Solar from-scratch?" → `solar-open-100b` → `[[from-scratch-debate]]` + `[[layernorm-fingerprint-fallacy]]` + `[[glm-family]]`.
3. **Cite using `[[page-title]]`.** Reserve plain markdown links for external sources only.
4. **Wiki not enough?** Say "위키에 없음 — 외부 자료 보강" explicitly. If the new content has lasting value, drop it under `wiki/raw/YYYY-MM-DD_<source>.md` and offer to ingest via `documentation:llm-wiki`.

## Quick Reference

| Need | Page(s) |
|------|---------|
| LLM-DNA core | `llm-dna-overview`, `dna-extraction-pipeline`, `random-projection`, `inheritance-and-determinism` |
| Tree construction | `phylogenetic-tree`, `neighbor-joining`, `upgma`, `distance-metrics` |
| Target models | `solar-open-100b`, `k-exaone-236b`, `ax-k1` |
| Reference families | `llama-3-family`, `qwen-25-family`, `glm-family`, `mixtral-8x7b`, `deepseek-v25` |
| Methodology | `tokenizer-analysis`, `weight-analysis`, `architecture-analysis`, `behavior-analysis`, `model-provenance-testing`, `layernorm-fingerprint-fallacy` |
| Infra | `sagemaker-spot-training`, `aws-cdk-typescript`, `huggingface-hub-usage` |
| Policy / context | `dokpamo-project`, `from-scratch-debate` |

## Q&A vs Wiki

The project runs both `docs/tutorial/` (chronological Q&A learning log) and `wiki/` (compiled best understanding). Flow on a new question: **wiki check → answer → tutorial Q&A append → wiki update if non-trivial**. Drives both forward without duplication.

## Content vs Operational — sister skill `llm-dna-extraction-playbook`

This skill is for **content** questions (what is X, where is Y in the wiki, how does NJ tree work). For **operational** work — submitting/debugging SageMaker training jobs, modifying `dna_train.py`/`submit_dna.py`/`analyze.py`, planning Phase 5/6 retries, cost analysis — defer to [`llm-dna-extraction-playbook`](../llm-dna-extraction-playbook/SKILL.md). That sister skill encodes 9 architectural lessons from Phase 4–6 ($34 learning curve cost), CLAUDE.md "인프라 운영 규칙" 9 rules, and entrypoint patches required for correct llm-dna 0.2.x behavior. Do not answer "should I run job X with config Y" from this skill alone — route through the playbook.

## Wiki Maintenance Triggers

Offer the user a wiki update when:
- A verification result is finalized (e.g., LLM-DNA tree analysis produces a conclusion)
- A new reference model joins the lineup
- A policy/definition shifts (e.g., "from-scratch" criteria)
- An external source is genuinely worth keeping (drop into `wiki/raw/`, then `llm-wiki ingest`)

## Common Mistakes

- Searching externally before reading `wiki/index.md` — ignores 28 pages of compiled knowledge
- Reading one page only, skipping `[[link]]` expansion — answers miss critical caveats (e.g., LayerNorm fingerprint fallacy)
- Editing `wiki/index.md` by hand — overwritten by the next `llm-wiki sync`
- Modifying files in `wiki/raw/` — raw zone is immutable; new info goes in a new file
- Concluding a verification but not updating the wiki — next session repeats the external lookup
- Citing a page whose frontmatter says `confidence: low` without flagging that uncertainty in the answer
