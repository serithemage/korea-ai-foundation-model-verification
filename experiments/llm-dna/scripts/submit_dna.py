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
from sagemaker.estimator import Estimator

REGION = "us-east-1"
STACK = "LlmDnaStack"
SCORE_THRESHOLD = 3  # below this, treat instance type as "no capacity"


def stack_outputs() -> dict[str, str]:
    cf = boto3.client("cloudformation", region_name=REGION)
    outs = cf.describe_stacks(StackName=STACK)["Stacks"][0]["Outputs"]
    return {o["OutputKey"]: o["OutputValue"] for o in outs}


_score_cache: dict[str, int] = {}


def spot_score(instance_type: str) -> int:
    """Cached EC2 Spot Placement Score for an ml.* instance type in our region."""
    if instance_type in _score_cache:
        return _score_cache[instance_type]
    ec2_type = instance_type[len("ml.") :] if instance_type.startswith("ml.") else instance_type
    try:
        ec2 = boto3.client("ec2", region_name=REGION)
        resp = ec2.get_spot_placement_scores(
            InstanceTypes=[ec2_type],
            TargetCapacity=1,
            SingleAvailabilityZone=True,
            RegionNames=[REGION],
        )
        score = max((s["Score"] for s in resp["SpotPlacementScores"]), default=0)
    except Exception:
        score = 0
    _score_cache[instance_type] = score
    return score


def maybe_upgrade_instance(item: dict[str, Any], fallback: str | None) -> tuple[str, bool]:
    """Return (instance_type, upgraded?). If fallback set and configured spot is starved,
    upgrade to fallback when its score is healthier."""
    cfg_inst = item["instance_type"]
    if not fallback:
        return cfg_inst, False
    cur_score = spot_score(cfg_inst)
    if cur_score >= SCORE_THRESHOLD:
        return cfg_inst, False
    fb_inst = fallback if fallback.startswith("ml.") else f"ml.{fallback}"
    fb_score = spot_score(fb_inst)
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
    fallback: str | None = None,
) -> None:
    model_id = item["id"]
    instance_type, upgraded = maybe_upgrade_instance(item, fallback)
    load_in = item.get("load_in", "fp16")
    trust_remote = item.get("trust_remote_code", True)

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
    print(f"  output       = {output_path}")
    if dry_run:
        print("  (dry-run, not submitted)\n")
        return

    estimator = Estimator(
        image_uri=outs["ContainerRepoUri"] + ":latest",
        role=outs["SageMakerRoleArn"],
        instance_count=1,
        instance_type=instance_type,
        hyperparameters=hp,
        environment={
            "HF_TOKEN_SECRET_ARN": outs["HfTokenSecretArn"],
            "S3_OUTPUT_PREFIX": f"s3://{outs['ArtifactsBucketName']}/dna/{job_name}/raw",
            "AWS_REGION": REGION,
        },
        use_spot_instances=True,
        max_run=7200,           # 2h compute cap
        max_wait=14400,         # 4h total wait (spot may queue)
        output_path=output_path,
        sagemaker_session=sess,
        tags=[
            {"Key": "project", "Value": "korea-ai-foundation-model-verification"},
            {"Key": "component", "Value": "llm-dna"},
            {"Key": "model", "Value": model_id},
        ],
        disable_profiler=True,  # Required for p5
    )

    estimator.fit(job_name=job_name, wait=False)
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
    args = p.parse_args()

    outs = stack_outputs()
    print("Stack outputs:")
    for k, v in outs.items():
        print(f"  {k} = {v}")
    print()

    cfg = yaml.safe_load(Path("configs/models.yaml").read_text())
    extraction = cfg.get("dna_extraction", {})

    sess = sagemaker.Session(boto3.Session(region_name=REGION))
    items = load_lineup(args.tier if args.tier != "all" else None, args.model)
    print(f"Submitting {len(items)} job(s) (dry_run={args.dry_run})\n")

    for item in items:
        submit_one(item, outs, sess, extraction, args.dry_run, args.fallback_to)
        if not args.dry_run:
            time.sleep(2)  # avoid TPS bursts when submitting many


if __name__ == "__main__":
    main()
