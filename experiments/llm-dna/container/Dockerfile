# Base: SageMaker PyTorch DLC (training image)
# us-east-1 SageMaker Deep Learning Container with PyTorch + CUDA + SageMaker training toolkit
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker-v1.22-2025-10-27-19-53-53

# Apt deps (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Python deps: llm-dna with full extras + boto3 (Secrets Manager) + hf_transfer (fast HF download)
# llm-dna[full] pulls in torch + transformers + bitsandbytes + GPTQ + compressed-tensors + ...
# We pin to known-working version range; loosen if upstream breaks.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        "llm-dna[quantization]>=0.1" \
        boto3 \
        hf_transfer \
        huggingface_hub

# Speed up HF downloads via Rust transfer crate
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV TRANSFORMERS_VERBOSITY=info

# Copy entrypoint
COPY src/ /opt/ml/code/
WORKDIR /opt/ml/code

# SageMaker training toolkit reads SAGEMAKER_PROGRAM env (set by Estimator) and runs it.
# We default to dna_train.py when running outside SageMaker.
ENV SAGEMAKER_PROGRAM=dna_train.py

ENTRYPOINT ["python", "/opt/ml/code/dna_train.py"]
