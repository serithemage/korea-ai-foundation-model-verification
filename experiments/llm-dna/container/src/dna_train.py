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

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("dna_train")


def read_hyperparameters() -> dict:
    """SageMaker writes hyperparameters to /opt/ml/input/config/hyperparameters.json."""
    path = Path("/opt/ml/input/config/hyperparameters.json")
    if path.exists():
        return json.loads(path.read_text())
    log.warning("No SageMaker hyperparameters file; falling back to env vars")
    return {}


def fetch_hf_token(secret_arn: str, region: str) -> str:
    client = boto3.client("secretsmanager", region_name=region)
    resp = client.get_secret_value(SecretId=secret_arn)
    return resp["SecretString"]


def upload_to_s3(local_path: Path, s3_uri: str) -> None:
    assert s3_uri.startswith("s3://"), f"bad s3 uri: {s3_uri}"
    bucket, _, key = s3_uri[5:].partition("/")
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    log.info("Uploaded %s -> %s", local_path, s3_uri)


def main() -> int:
    hp = read_hyperparameters()
    model_name = hp.get("model_name") or os.environ.get("MODEL_NAME")
    if not model_name:
        log.error("model_name hyperparameter is required")
        return 2

    dataset = hp.get("dataset", "rand")
    max_samples = int(hp.get("max_samples", 100))
    dna_dim = int(hp.get("dna_dim", 128))
    reduction = hp.get("reduction", "random_projection")
    random_seed = int(hp.get("random_seed", 42))
    load_in = hp.get("load_in", "fp16")
    trust_remote = str(hp.get("trust_remote", "true")).lower() == "true"

    region = os.environ.get("AWS_REGION", "us-east-1")
    secret_arn = os.environ.get("HF_TOKEN_SECRET_ARN")
    s3_output_prefix = os.environ.get("S3_OUTPUT_PREFIX", "")

    # 1. Fetch HF token from Secrets Manager
    if secret_arn:
        log.info("Fetching HF token from Secrets Manager: %s", secret_arn)
        os.environ["HF_TOKEN"] = fetch_hf_token(secret_arn, region)
    else:
        log.warning("HF_TOKEN_SECRET_ARN not set; relying on existing HF_TOKEN env var")

    # 2. Build calc-dna CLI args
    output_dir = Path("/opt/ml/model")  # SageMaker auto-uploads /opt/ml/model to S3
    output_dir.mkdir(parents=True, exist_ok=True)

    args = [
        "calc-dna",
        "--model-name", model_name,
        "--dataset", dataset,
        "--max-samples", str(max_samples),
        "--dna-dim", str(dna_dim),
        "--reduction-method", reduction,
        "--random-seed", str(random_seed),
        "--device", "cuda",
        "--output-dir", str(output_dir),
        "--continue-on-error",
    ]
    if trust_remote:
        args.append("--trust-remote-code")
    if load_in == "8bit":
        args.append("--load-in-8bit")
    elif load_in == "4bit":
        args.append("--load-in-4bit")
    # fp16 is default; no flag needed

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
