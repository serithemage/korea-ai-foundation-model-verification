#!/usr/bin/env python3
"""Pull DNA vectors from S3 and build a phylogenetic tree of LLMs.

After Phase 4/5 SageMaker jobs complete, run:
    python scripts/analyze.py                       # cosine + euclidean, no bootstrap
    python scripts/analyze.py --bootstrap 100       # 100 dimension-resamples per metric
    python scripts/analyze.py --metric cosine

Outputs (per metric):
    out/lineage_<metric>.nwk            Newick (open in iTOL or ete3)
    out/distance_<metric>.csv           NxN distance matrix
    out/distance_<metric>.png           Heatmap visualization
    out/lineage_<metric>.png            matplotlib tree (when bootstrap given,
                                        clade labels include support %)
    out/lineage_<metric>_ascii.txt      ASCII tree dump
"""

from __future__ import annotations

import argparse
import io
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import boto3
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from Bio.Phylo.TreeConstruction import DistanceMatrix, DistanceTreeConstructor
from Bio import Phylo
from scipy.spatial.distance import pdist, squareform

REGION = "us-east-1"
STACK = "LlmDnaStack"
OUT_DIR = Path("out")

# Color groups for visualization
COLOR_GROUPS = {
    "korean-target": ("#d62728", ["upstage/", "LGAI-EXAONE/", "skt/A.X"]),
    "llama":         ("#1f77b4", ["meta-llama/"]),
    "qwen":          ("#2ca02c", ["Qwen/"]),
    "glm":           ("#9467bd", ["zai-org/", "THUDM/"]),
    "mistral":       ("#ff7f0e", ["mistralai/"]),
    "deepseek":      ("#8c564b", ["deepseek-ai/"]),
    "other":         ("#7f7f7f", []),
}


def color_for(model: str) -> str:
    for _, (color, prefixes) in COLOR_GROUPS.items():
        if any(model.startswith(p) for p in prefixes):
            return color
    return COLOR_GROUPS["other"][0]


def stack_outputs() -> dict[str, str]:
    cf = boto3.client("cloudformation", region_name=REGION)
    outs = cf.describe_stacks(StackName=STACK)["Stacks"][0]["Outputs"]
    return {o["OutputKey"]: o["OutputValue"] for o in outs}


def list_dna_artifacts(bucket: str) -> dict[str, str]:
    """Return {model_id: s3_key_to_npy_or_json}."""
    s3 = boto3.client("s3", region_name=REGION)
    paginator = s3.get_paginator("list_objects_v2")
    artifacts: dict[str, str] = {}
    pat = re.compile(r"dna/(?P<job>[^/]+)/raw/(?P<model>[^/]+)/(?P<file>.+\.(npy|json))$")
    for page in paginator.paginate(Bucket=bucket, Prefix="dna/"):
        for obj in page.get("Contents", []):
            m = pat.match(obj["Key"])
            if not m:
                continue
            model = m.group("model").replace("__", "/")
            # Prefer .npy over .json
            if obj["Key"].endswith(".npy") or model not in artifacts:
                artifacts[model] = obj["Key"]
    return artifacts


def fetch_dna_vector(bucket: str, key: str) -> np.ndarray:
    s3 = boto3.client("s3", region_name=REGION)
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if key.endswith(".npy"):
        return np.load(io.BytesIO(body))
    # JSON fallback
    import json
    obj = json.loads(body)
    return np.asarray(obj["vector"] if isinstance(obj, dict) else obj, dtype=np.float64)


def build_tree(vectors: dict[str, np.ndarray], metric: str) -> tuple[DistanceMatrix, "Phylo.BaseTree.Tree"]:
    names = sorted(vectors)
    matrix = np.stack([vectors[n] for n in names])
    condensed = pdist(matrix, metric=metric)
    full = squareform(condensed)
    # BioPython expects lower-triangular (incl diagonal) as list of lists
    lower = [list(full[i, : i + 1]) for i in range(len(names))]
    dm = DistanceMatrix(names=names, matrix=lower)
    tree = DistanceTreeConstructor().nj(dm)
    return dm, tree


def save_distance_csv(dm: DistanceMatrix, path: Path) -> None:
    n = len(dm.names)
    rows = ["," + ",".join(dm.names)]
    for i, name in enumerate(dm.names):
        cells = []
        for j in range(n):
            if j <= i:
                cells.append(f"{dm[i, j]:.6f}")
            else:
                cells.append(f"{dm[j, i]:.6f}")
        rows.append(name + "," + ",".join(cells))
    path.write_text("\n".join(rows))


def plot_heatmap(dm: DistanceMatrix, path: Path, metric: str) -> None:
    n = len(dm.names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = dm[i, j] if j <= i else dm[j, i]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * n + 4), max(7, 0.5 * n + 3)))
    im = ax.imshow(M, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(dm.names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(dm.names, fontsize=8)
    for i, name in enumerate(dm.names):
        c = color_for(name)
        ax.get_xticklabels()[i].set_color(c)
        ax.get_yticklabels()[i].set_color(c)
    ax.set_title(f"Pairwise {metric} distance")
    fig.colorbar(im, ax=ax, label=metric)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_tree(tree, path: Path, metric: str, support: dict[frozenset, float] | None = None) -> None:
    if support:
        for clade in tree.get_nonterminals():
            taxa = frozenset(t.name for t in clade.get_terminals())
            if taxa in support:
                clade.confidence = round(support[taxa] * 100)
    fig = plt.figure(figsize=(10, max(6, 0.4 * len(tree.get_terminals()) + 2)))
    ax = fig.add_subplot(1, 1, 1)
    Phylo.draw(
        tree,
        axes=ax,
        do_show=False,
        label_colors=lambda name: color_for(name) if name else "#000",
        branch_labels=(lambda c: f"{int(c.confidence)}" if c.confidence else None) if support else None,
    )
    ax.set_title(f"NJ tree ({metric}{', bootstrap support shown' if support else ''})")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def bootstrap_clades(
    vectors: dict[str, np.ndarray],
    metric: str,
    n_iter: int,
    rng: np.random.Generator,
) -> dict[frozenset, float]:
    """Return per-clade support fraction across n_iter dimension-resamples."""
    names = sorted(vectors)
    matrix = np.stack([vectors[n] for n in names])
    d = matrix.shape[1]
    keep_n = max(int(d * 0.8), 8)  # 80% dim subsample per replicate
    counter: Counter[frozenset] = Counter()
    for _ in range(n_iter):
        idx = rng.choice(d, size=keep_n, replace=False)
        sub = matrix[:, idx]
        condensed = pdist(sub, metric=metric)
        full = squareform(condensed)
        lower = [list(full[i, : i + 1]) for i in range(len(names))]
        dm = DistanceMatrix(names=names, matrix=lower)
        tree = DistanceTreeConstructor().nj(dm)
        for clade in tree.get_nonterminals():
            taxa = frozenset(t.name for t in clade.get_terminals())
            if 1 < len(taxa) < len(names):
                counter[taxa] += 1
    return {taxa: cnt / n_iter for taxa, cnt in counter.items()}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--metric", action="append", default=None, help="Distance metric; repeatable")
    p.add_argument("--bucket", help="Override S3 bucket (default: from CDK stack outputs)")
    p.add_argument("--bootstrap", type=int, default=0, help="Bootstrap iterations (0 = off)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    metrics = args.metric or ["cosine", "euclidean"]

    OUT_DIR.mkdir(exist_ok=True)
    bucket = args.bucket or stack_outputs()["ArtifactsBucketName"]
    print(f"S3 bucket: {bucket}")

    artifacts = list_dna_artifacts(bucket)
    if not artifacts:
        print("No DNA artifacts found yet.")
        return 1
    print(f"Found {len(artifacts)} model artifacts")

    vectors: dict[str, np.ndarray] = {}
    for model, key in sorted(artifacts.items()):
        v = fetch_dna_vector(bucket, key)
        vectors[model] = v
        print(f"  {model}: shape={v.shape}, key={key}")

    rng = np.random.default_rng(args.seed)
    for metric in metrics:
        dm, tree = build_tree(vectors, metric)
        nwk = OUT_DIR / f"lineage_{metric}.nwk"
        Phylo.write(tree, str(nwk), "newick")
        save_distance_csv(dm, OUT_DIR / f"distance_{metric}.csv")
        plot_heatmap(dm, OUT_DIR / f"distance_{metric}.png", metric)

        support = None
        if args.bootstrap > 0:
            print(f"  Bootstrap {args.bootstrap}× for {metric} ...")
            support = bootstrap_clades(vectors, metric, args.bootstrap, rng)

        plot_tree(tree, OUT_DIR / f"lineage_{metric}.png", metric, support)

        ascii_path = OUT_DIR / f"lineage_{metric}_ascii.txt"
        with ascii_path.open("w") as f:
            Phylo.draw_ascii(tree, file=f)

        print(f"\n=== {metric} ===")
        print(f"  Newick     : {nwk}")
        print(f"  Distance   : {OUT_DIR}/distance_{metric}.{{csv,png}}")
        print(f"  Tree       : {OUT_DIR}/lineage_{metric}.png")
        print(f"  ASCII tree :")
        Phylo.draw_ascii(tree)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
