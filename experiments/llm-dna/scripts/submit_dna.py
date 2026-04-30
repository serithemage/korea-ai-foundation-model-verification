#!/usr/bin/env python3
"""Submit SageMaker training jobs for LLM-DNA extraction.

Reads model lineup from configs/models.yaml, looks up the deployed CDK stack
outputs (S3 bucket, ECR image, secret ARN, role ARN), and submits one
training job per model.

Usage:
    python scripts/submit_dna.py --tier reference   # all reference models
    python scripts/submit_dna.py --tier target      # all 3 Korean targets
    python scripts/submit_dna.py --model upstage/Solar-Open-100B
    python scripts/submit_dna.py --tier reference --dry-run

Capacity fallback:
    --fallback-to p5.48xlarge   if the configured ml.p5.4xlarge spot has score<3,
                                upgrade to ml.p5.48xlarge automatically (more $$ but
                                avoids 4h Insufficient-Capacity wait).
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
import sagemaker
import yaml
from sagemaker.pytorch import PyTorch

DEFAULT_REGION = "us-east-1"
SCORE_THRESHOLD = 3  # below this, treat instance type as "no capacity"


def stack_name(region: str) -> str:
    """Match the naming used in cdk/bin/llm-dna.ts (us-east-1 keeps the short name)."""
    return "LlmDnaStack" if region == "us-east-1" else f"LlmDnaStack-{region}"

# SageMaker Deep Learning Container — pre-cached on every spot worker host.
# Matches what works in roboco-io/research/serverless-autoresearch (proven on p5.4xl spot).
DLC_FRAMEWORK_VERSION = "2.8.0"
DLC_PY_VERSION = "py312"

# Local directory uploaded to S3 by SageMaker each job (entrypoint + requirements.txt)
SOURCE_DIR = Path("source")
ENTRY_POINT = "dna_train.py"


def stack_outputs(region: str) -> dict[str, str]:
    cf = boto3.client("cloudformation", region_name=region)
    outs = cf.describe_stacks(StackName=stack_name(region))["Stacks"][0]["Outputs"]
    return {o["OutputKey"]: o["OutputValue"] for o in outs}


_score_cache: dict[tuple[str, str], int] = {}


def spot_score(instance_type: str, region: str) -> int:
    """Cached EC2 Spot Placement Score for an ml.* instance type in `region`."""
    key = (region, instance_type)
    if key in _score_cache:
        return _score_cache[key]
    ec2_type = instance_type[len("ml.") :] if instance_type.startswith("ml.") else instance_type
    try:
        ec2 = boto3.client("ec2", region_name=region)
        resp = ec2.get_spot_placement_scores(
            InstanceTypes=[ec2_type],
            TargetCapacity=1,
            SingleAvailabilityZone=True,
            RegionNames=[region],
        )
        score = max((s["Score"] for s in resp["SpotPlacementScores"]), default=0)
    except Exception:
        score = 0
    _score_cache[key] = score
    return score


def maybe_upgrade_instance(item: dict[str, Any], fallback: str | None, region: str) -> tuple[str, bool]:
    """Return (instance_type, upgraded?). If fallback set and configured spot is starved,
    upgrade to fallback when its score is healthier."""
    cfg_inst = item["instance_type"]
    if not fallback:
        return cfg_inst, False
    cur_score = spot_score(cfg_inst, region)
    if cur_score >= SCORE_THRESHOLD:
        return cfg_inst, False
    fb_inst = fallback if fallback.startswith("ml.") else f"ml.{fallback}"
    fb_score = spot_score(fb_inst, region)
    if fb_score > cur_score:
        return fb_inst, True
    return cfg_inst, False


def load_lineup(tier: str | None, model: str | None) -> list[dict[str, Any]]:
    cfg = yaml.safe_load(Path("configs/models.yaml").read_text())
    items: list[dict[str, Any]] = []
    if model:
        for group in ("targets", "references"):
            blob = cfg.get(group, {})
            if isinstance(blob, list):
                items.extend(m for m in blob if m["id"] == model)
            elif isinstance(blob, dict):
                for sub in blob.values():
                    items.extend(m for m in sub if m["id"] == model)
        if not items:
            raise SystemExit(f"Model {model!r} not in configs/models.yaml")
        return items

    if tier in (None, "target", "targets"):
        items.extend(cfg.get("targets", []))
    if tier in (None, "reference", "references"):
        for sub in cfg.get("references", {}).values():
            items.extend(sub)
    return items


def make_job_name(model_id: str) -> str:
    safe = model_id.replace("/", "-").replace("_", "-").replace(".", "-").lower()
    safe = "".join(c if c.isalnum() or c == "-" else "-" for c in safe)[:40].strip("-")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"llmdna-{safe}-{ts}"


def submit_one(
    item: dict[str, Any],
    outs: dict[str, str],
    sess: sagemaker.Session,
    extraction: dict[str, Any],
    dry_run: bool,
    region: str,
    fallback: str | None = None,
    instance_override: str | None = None,
    load_in_override: str | None = None,
    max_wait_hours: int = 4,
) -> None:
    model_id = item["id"]
    if instance_override:
        instance_type, upgraded = instance_override, False
    else:
        instance_type, upgraded = maybe_upgrade_instance(item, fallback, region)
    load_in = load_in_override or item.get("load_in", "fp16")
    trust_remote = item.get("trust_remote_code", True)

    # volume_size controls the EBS volume mounted at /opt/ml/{output,errors,input,model}.
    # NOT used for HF cache (entrypoint redirects HF_HOME to /tmp on the 1.7 TB instance
    # store). /opt/ml/output is what gets uploaded to S3 — keep it small to avoid S3 cost.
    # 30 GB default is fine; bump only if the model output (DNA + responses) might be huge.
    volume_size = 30

    hp = {
        "model_name": model_id,
        "dataset": extraction.get("dataset", "rand"),
        "max_samples": extraction.get("max_samples", 100),
        "dna_dim": extraction.get("dna_dim", 128),
        "reduction": extraction.get("reduction_method", "random_projection"),
        "random_seed": extraction.get("random_seed", 42),
        "load_in": load_in,
        "trust_remote": str(trust_remote).lower(),
    }

    job_name = make_job_name(model_id)
    output_path = f"s3://{outs['ArtifactsBucketName']}/dna/{job_name}/"

    print(f"[{job_name}]")
    print(f"  model        = {model_id}")
    if upgraded:
        cur_score = _score_cache.get(item["instance_type"], "?")
        new_score = _score_cache.get(instance_type, "?")
        print(f"  instance     = {instance_type}  (UPGRADED from {item['instance_type']}, "
              f"score {cur_score} → {new_score})")
    else:
        print(f"  instance     = {instance_type}")
    print(f"  load_in      = {load_in}")
    print(f"  volume_size  = {volume_size}GB")
    print(f"  output       = {output_path}")
    if dry_run:
        print("  (dry-run, not submitted)\n")
        return

    estimator = PyTorch(
        # SageMaker DLC (pre-cached on every spot worker — fast cold start, less spot reclaim risk
        # than 10GB+ custom ECR images that have to pull cross-AZ on each launch).
        framework_version=DLC_FRAMEWORK_VERSION,
        py_version=DLC_PY_VERSION,
        source_dir=str(SOURCE_DIR),     # uploaded to S3 by SageMaker; requirements.txt auto-installed
        entry_point=ENTRY_POINT,
        role=outs["SageMakerRoleArn"],
        instance_count=1,
        instance_type=instance_type,
        hyperparameters=hp,
        environment={
            "HF_TOKEN_SECRET_ARN": outs["HfTokenSecretArn"],
            "S3_OUTPUT_PREFIX": f"s3://{outs['ArtifactsBucketName']}/dna/{job_name}/raw",
            "AWS_REGION": region,
            # hf_transfer = parallel downloader. Past benefit was throughput, but for big
            # models it pre-allocates many shard temp files concurrently, transiently using
            # ~2x disk. We've seen 100B+ models fail with No space left even on 305 GB EBS.
            # Sequential downloads are slower but bounded.
            "HF_HUB_ENABLE_HF_TRANSFER": "0",
        },
        use_spot_instances=True,
        # 8 h compute cap: 70B+ models on g7e.12xl spot can hit 1 interruption mid-run,
        # losing ~2 h of progress, then need another full 2.5 h for the second attempt.
        # 4 h was tight for 100 prompts × 85 s/prompt + restart overhead. Spot still bills
        # only actual compute time used.
        max_run=28800,
        max_wait=max_wait_hours * 3600,     # total wait incl. spot queueing
        volume_size=volume_size,            # EBS GB; default 30 is too small for big HF models
        output_path=output_path,
        sagemaker_session=sess,
        tags=[
            {"Key": "project", "Value": "korea-ai-foundation-model-verification"},
            {"Key": "component", "Value": "llm-dna"},
            {"Key": "model", "Value": model_id},
        ],
        disable_profiler=True,  # Required for p5
    )

    estimator.fit(job_name=job_name, wait=False, logs=False)
    print(f"  submitted    = {job_name}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tier", choices=["target", "reference", "all"], default="all")
    p.add_argument("--model", help="Submit a single model by id")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--fallback-to",
        default=None,
        help="Auto-upgrade instance if configured spot has score<3 (e.g. p5.48xlarge)",
    )
    p.add_argument(
        "--instance",
        default=None,
        help="Force a specific instance type (e.g. ml.g6.2xlarge), overrides yaml & --fallback-to",
    )
    p.add_argument(
        "--load-in",
        choices=["fp16", "8bit", "4bit"],
        default=None,
        help="Override the load-in precision from yaml",
    )
    p.add_argument(
        "--max-wait-hours",
        type=int,
        default=4,
        help="SageMaker max_wait in hours (default 4); raise when spot capacity is tight",
    )
    p.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=f"AWS region (default {DEFAULT_REGION}). Stack name auto-derived: us-east-1 → "
             f"LlmDnaStack, others → LlmDnaStack-<region>",
    )
    args = p.parse_args()

    print(f"Region: {args.region}  Stack: {stack_name(args.region)}\n")
    outs = stack_outputs(args.region)
    print("Stack outputs:")
    for k, v in outs.items():
        print(f"  {k} = {v}")
    print()

    cfg = yaml.safe_load(Path("configs/models.yaml").read_text())
    extraction = cfg.get("dna_extraction", {})

    sess = sagemaker.Session(boto3.Session(region_name=args.region))
    items = load_lineup(args.tier if args.tier != "all" else None, args.model)
    print(f"Submitting {len(items)} job(s) (dry_run={args.dry_run})\n")

    for item in items:
        submit_one(
            item, outs, sess, extraction, args.dry_run,
            region=args.region,
            fallback=args.fallback_to,
            instance_override=args.instance,
            load_in_override=args.load_in,
            max_wait_hours=args.max_wait_hours,
        )
        if not args.dry_run:
            time.sleep(2)  # avoid TPS bursts when submitting many


if __name__ == "__main__":
    main()
