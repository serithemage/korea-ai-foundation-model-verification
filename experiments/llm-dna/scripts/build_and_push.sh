#!/usr/bin/env bash
# Build the llm-dna container for linux/amd64 (SageMaker target) and push to ECR.
# Usage: ./scripts/build_and_push.sh [tag]   (default tag: latest)
set -euo pipefail

REGION="${AWS_REGION:-us-east-1}"
ACCOUNT="${AWS_ACCOUNT_ID:-779411790546}"
REPO_URI="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/llm-dna"
TAG="${1:-latest}"
IMAGE_URI="${REPO_URI}:${TAG}"

# SageMaker DLC base lives in account 763104351884 — need a separate ECR login for the pull.
SAGEMAKER_DLC_REGISTRY="763104351884.dkr.ecr.${REGION}.amazonaws.com"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_DIR="${ROOT}/container"

echo "==> Logging in to our ECR repo (${ACCOUNT})"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

echo "==> Logging in to SageMaker DLC registry (${SAGEMAKER_DLC_REGISTRY})"
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${SAGEMAKER_DLC_REGISTRY}"

echo "==> Ensuring buildx builder exists"
docker buildx inspect llm-dna-builder >/dev/null 2>&1 \
  || docker buildx create --name llm-dna-builder --use --bootstrap

echo "==> Building & pushing ${IMAGE_URI} (linux/amd64)"
docker buildx build \
  --platform linux/amd64 \
  --tag "${IMAGE_URI}" \
  --push \
  --provenance=false \
  --progress=plain \
  "${CONTAINER_DIR}"

echo "==> Verifying image in ECR"
aws ecr describe-images --repository-name llm-dna --region "${REGION}" \
  --query "imageDetails[?contains(imageTags, '${TAG}')].{Tags:imageTags,Pushed:imagePushedAt,SizeMB:to_number(imageSizeInBytes)}"  \
  --output table

echo ""
echo "Done. Image URI: ${IMAGE_URI}"
