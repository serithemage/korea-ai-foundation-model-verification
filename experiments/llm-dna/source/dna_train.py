#!/usr/bin/env python3
"""SageMaker training-job entrypoint for LLM-DNA extraction.

Reads model and config from SageMaker hyperparameters / env vars,
fetches HF token from Secrets Manager, runs llm-dna `calc-dna`,
and uploads the resulting DNA vector + metadata to S3.

Hyperparameters (set via Estimator):
    model_name      HuggingFace model id, e.g. upstage/Solar-Open-100B
    dataset         "rand" (default) or other llm-dna dataset
    max_samples     int, default 100
    dna_dim         int, default 128
    reduction       "random_projection" | "pca" | "svd"
    random_seed     int, default 42
    load_in         "fp16" | "8bit" | "4bit"
    trust_remote    "true" | "false"

Env vars (set via Estimator.environment):
    HF_TOKEN_SECRET_ARN  ARN of Secrets Manager secret holding the HF token
    S3_OUTPUT_PREFIX     e.g. s3://bucket/dna/
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dna_train")


def parse_args() -> argparse.Namespace:
    """SageMaker passes hyperparameters as `--key value` CLI args.

    NOTE: don't read hyperparameters.json directly — SageMaker writes each value
    as a JSON-encoded string, so plain json.loads of the file leaves embedded
    quotes around string values (e.g. 'random_projection' becomes
    '"random_projection"'), which then fails calc-dna's choice validation.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", default="rand")
    p.add_argument("--max_samples", type=int, default=100)
    p.add_argument("--dna_dim", type=int, default=128)
    p.add_argument("--reduction", default="random_projection")
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--load_in", default="fp16", choices=["fp16", "8bit", "4bit"])
    p.add_argument("--trust_remote", default="true")
    return p.parse_args()


def fetch_hf_token(secret_arn: str, region: str) -> str:
    """Fetch HF token from Secrets Manager, defensively stripping accidental quotes.

    Past incident: secret was stored as ``'hf_xxx'`` (with literal single quotes)
    via shell quoting confusion. HF auth rejects the quoted form as 401 Invalid,
    but anonymous/non-gated downloads still work — making the bug invisible until
    a gated model (Llama) is hit.
    """
    client = boto3.client("secretsmanager", region_name=region)
    resp = client.get_secret_value(SecretId=secret_arn)
    return resp["SecretString"].strip().strip("'\"")


def upload_to_s3(local_path: Path, s3_uri: str) -> None:
    assert s3_uri.startswith("s3://"), f"bad s3 uri: {s3_uri}"
    bucket, _, key = s3_uri[5:].partition("/")
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    log.info("Uploaded %s -> %s", local_path, s3_uri)


def patch_model_wrapper_for_multi_gpu() -> None:
    """Patch the installed llm_dna ModelWrapper so non-quantized large models
    use accelerate's auto multi-GPU dispatch instead of single-GPU placement.

    Upstream behavior (llm_dna 0.1.x ModelWrapper.py, non-quantized branch):
        model_kwargs["device_map"] = {"" : self.device}     # always single GPU
    This pins the entire model to cuda:0 even on multi-GPU instances
    (p5.48xl 8xH100, g7e.48xl 8xL40S), making 100B+ fp16 weights unfittable.

    Patch: when torch.cuda.device_count() > 1, switch to device_map="auto" so
    accelerate distributes layers across all visible GPUs, and turn on
    low_cpu_mem_usage to avoid 2x peak host memory during shard load.
    """
    import importlib.util
    spec = importlib.util.find_spec("llm_dna.models.ModelWrapper")
    if spec is None or spec.origin is None:
        log.warning("llm_dna.models.ModelWrapper not found; skipping patch")
        return
    fp = Path(spec.origin)
    src = fp.read_text()
    needle = 'model_kwargs["device_map"] = {"" : self.device}'
    if needle not in src:
        log.warning("ModelWrapper patch target not found (already patched or upstream changed?)")
        return
    replacement = (
        'if torch.cuda.is_available() and torch.cuda.device_count() > 1:\n'
        '                model_kwargs["device_map"] = "auto"\n'
        '                model_kwargs["low_cpu_mem_usage"] = True\n'
        '                self.logger.info("Patched: non-quantized model uses device_map=auto across %d GPUs" % torch.cuda.device_count())\n'
        '            else:\n'
        '                model_kwargs["device_map"] = {"" : self.device}'
    )
    src = src.replace(needle, replacement)
    fp.write_text(src)
    log.info("Patched ModelWrapper at %s for multi-GPU non-quantized loading", fp)


def patch_model_wrapper_for_min_new_tokens(max_cap: int | None = None) -> None:
    """Force base (non-instruct) models to generate non-empty responses,
    optionally capping the upper bound for speed.

    Symptom (2026-04-30, Mixtral-8x7B-v0.1 + Solar-Open-100B): both produced
    100/100 empty responses on rand AND squad probe sets. llm-dna's extraction
    is `text_response_embeddings_random_projection_concat` — empty text
    collapses every model to the same fallback embedding, faking a 0.0 distance
    pair. Other base models (Llama-3.1-8B, Qwen-7B) generated text fine, but
    Mixtral-base and Solar-base hit EOS on the very first generated token.

    Patch: inject `min_new_tokens=50` into the generation_kwargs so the model
    must emit at least 50 tokens regardless of EOS — produces a non-trivial
    text whose embedding becomes a meaningful fingerprint.

    Optional max_cap: also clamp `max_new_tokens` to at most max_cap. Needed
    for Solar-Open-100B which keeps generating up to the 2048 limit on every
    prompt (200s/prompt on p5.48xl); capping to 256 brings it to ~30-40s/prompt
    while still emitting enough text for a meaningful sentence-embedding
    fingerprint.
    """
    import importlib.util
    spec = importlib.util.find_spec("llm_dna.models.ModelWrapper")
    if spec is None or spec.origin is None:
        log.warning("llm_dna.models.ModelWrapper not found; skipping min_new_tokens patch")
        return
    fp = Path(spec.origin)
    src = fp.read_text()
    needle = '"max_new_tokens": max_new_tokens,  # Dynamically calculated to be safe'
    if needle not in src:
        log.warning("min_new_tokens patch target not found (already patched or upstream changed?)")
        return
    if max_cap:
        replacement = (
            f'"max_new_tokens": min({max_cap}, max_new_tokens),  # Patched: cap for speed\n'
            '                "min_new_tokens": min(50, max(1, max_new_tokens)),  # Patched: force base models to emit text'
        )
    else:
        replacement = (
            '"max_new_tokens": max_new_tokens,  # Dynamically calculated to be safe\n'
            '                "min_new_tokens": min(50, max(1, max_new_tokens)),  # Patched: force base models to emit text'
        )
    src = src.replace(needle, replacement)
    fp.write_text(src)
    log.info("Patched ModelWrapper at %s for min_new_tokens=50%s", fp,
             f", max_new_tokens cap={max_cap}" if max_cap else "")


def main() -> int:
    a = parse_args()
    model_name = a.model_name
    dataset = a.dataset
    max_samples = a.max_samples
    dna_dim = a.dna_dim
    reduction = a.reduction
    random_seed = a.random_seed
    load_in = a.load_in
    trust_remote = a.trust_remote.lower() == "true"

    region = os.environ.get("AWS_REGION", "us-east-1")
    secret_arn = os.environ.get("HF_TOKEN_SECRET_ARN")
    s3_output_prefix = os.environ.get("S3_OUTPUT_PREFIX", "")

    # 0. Redirect HF cache to /tmp on the instance store SSD.
    #
    # Discovered via `df -h` diagnostic 2026-04-28:
    #   /dev/nvme2n1   1.7T   /tmp                                          ← instance store
    #   /dev/nvme2n1         /opt/ml/output, /errors, /input, /model        ← also instance store
    #   /dev/nvme1n1   120G   /                                             ← root (small!)
    #
    # SageMaker only mounts /opt/ml/{output,errors,input,model} on the configured EBS volume
    # (or the much larger instance-store NVMe). Everything else under /opt/ml/, including
    # our /opt/ml/.cache/huggingface, lives on the 120 GB root volume that the DLC already
    # half-fills with python + cuda libs. Past failures (Mixtral 47 B on 335 GB EBS,
    # K-EXAONE 236 B on 640 GB EBS) all ran out at ~80-100 GB disk usage because we were
    # never actually using the configured EBS volume — we were filling up the root.
    #
    # /tmp is on the big NVMe and is ephemeral (good — no S3 upload of giant cache at end).
    hf_cache_root = Path("/tmp/.cache/huggingface")
    hf_cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_cache_root))
    os.environ.setdefault("HF_HUB_CACHE", str(hf_cache_root / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_cache_root))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(hf_cache_root / "sentence-transformers"))
    # hf-xet stores both chunk-cache AND assembled blobs, effectively doubling disk usage
    # for the same model. For 100B+ models this is the difference between fitting in 200 GB
    # vs blowing past 400 GB. Force traditional single-copy download.
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # Even with xet disabled, hf_transfer's parallel downloader can pre-allocate temp files
    # for many shards simultaneously, transiently doubling disk pressure. Use sequential
    # downloads (slower but bounded) and rely on aria2c-like behavior only when needed.
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    log.info("Redirecting HF caches under %s (EBS volume), xet+hf_transfer disabled",
             hf_cache_root)

    # Diagnostic: dump disk layout before model download. Helps spot whether something
    # (DLC libs, hf-xet leftover, snapshot copies) is silently consuming the EBS volume.
    try:
        log.info("--- df -h ---\n%s", subprocess.check_output(["df", "-h"], text=True))
        log.info("--- mount ---\n%s", subprocess.check_output(["mount"], text=True))
    except Exception as e:
        log.warning("disk diagnostic failed: %s", e)

    # 1. Fetch HF token from Secrets Manager
    if secret_arn:
        log.info("Fetching HF token from Secrets Manager: %s", secret_arn)
        os.environ["HF_TOKEN"] = fetch_hf_token(secret_arn, region)
    else:
        log.warning("HF_TOKEN_SECRET_ARN not set; relying on existing HF_TOKEN env var")

    # 2. Build calc-dna CLI args
    output_dir = Path("/opt/ml/model")  # SageMaker auto-uploads /opt/ml/model to S3
    output_dir.mkdir(parents=True, exist_ok=True)

    # max-length cap: bitsandbytes 4-bit auto device_map sizing reserves space for KV cache
    # at the model's default max_position_embeddings. Solar-100B / K-EXAONE etc. have 32 K +
    # context windows, which inflates activation memory enormously and triggers spurious
    # CPU offload. We only need ~512-1024 tokens for the rand probe set; cap at 2048.
    args = [
        "calc-dna",
        "--model-name", model_name,
        "--dataset", dataset,
        "--max-samples", str(max_samples),
        "--dna-dim", str(dna_dim),
        "--reduction-method", reduction,
        "--random-seed", str(random_seed),
        # `cuda` pins llm-dna to GPU 0 only (single device). For 100B+ models on multi-GPU
        # instances (p5.48xl 8×H100, g7e.48xl 8×L40S) we need to use them all → `auto`
        # lets transformers' device_map="auto" balance across every visible GPU.
        "--device", "auto",
        "--output-dir", str(output_dir),
        "--max-length", "2048",
        "--continue-on-error",
    ]
    if trust_remote:
        args.append("--trust-remote-code")
    if load_in == "8bit":
        args.append("--load-in-8bit")
    elif load_in == "4bit":
        args.append("--load-in-4bit")
    else:
        # fp16: must explicitly disable quantization. Otherwise calc-dna
        # auto-enables 8-bit for any model >=7B (extraction.py:435-437),
        # which then hits ModelWrapper's single-GPU pin + bitsandbytes
        # CPU-offload refusal — exact failure mode seen on K-EXAONE/Solar
        # 2026-04-29 jobs.
        args.append("--no-quantization")

    # Patch the installed llm_dna ModelWrapper so the non-quantized branch
    # uses accelerate auto device_map across all GPUs. Required for
    # 100B+/200B+ fp16 weights that don't fit on a single H100/L40S.
    patch_model_wrapper_for_multi_gpu()
    # Force base models (Mixtral-base, Solar-base) to emit at least 50 tokens.
    # Without this, llm-dna's text_response_embeddings_random_projection_concat
    # collapses every empty-response model to the same fallback fingerprint.
    # For Solar-Open-100B specifically, also cap max_new_tokens — without a cap
    # this model fills the full 2048 limit on every prompt and runs at
    # ~200s/prompt on p5.48xl multi-GPU. Capping to 256 keeps the fingerprint
    # text long enough for the sentence-encoder while restoring sane wall time.
    max_cap = 256 if model_name == "upstage/Solar-Open-100B" else None
    patch_model_wrapper_for_min_new_tokens(max_cap=max_cap)

    log.info("Running: %s", " ".join(args))
    proc = subprocess.run(args, check=False)
    if proc.returncode != 0:
        log.error("calc-dna exited with code %d", proc.returncode)
        return proc.returncode

    # 3. Optional explicit S3 upload (in addition to SageMaker auto-upload of /opt/ml/model)
    if s3_output_prefix:
        # Copy DNA artifacts under output_dir to a flatter prefix per model
        safe_id = model_name.replace("/", "__")
        for src in output_dir.rglob("*"):
            if src.is_file():
                rel = src.relative_to(output_dir)
                dst = f"{s3_output_prefix.rstrip('/')}/{safe_id}/{rel}"
                upload_to_s3(src, dst)

    log.info("DNA extraction complete for %s", model_name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
