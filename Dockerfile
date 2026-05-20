# ─────────────────────────────────────────────────────────────────────────────
#  Dockerfile
#  Base image : PyTorch 2.2 + CUDA 12.1 (GPU) or CPU-only
#  Falls back to CPU automatically if no GPU is available.
# ─────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

# Metadata
LABEL maintainer="FineTune-LoRA Project"
LABEL description="LoRA fine-tuning of a small causal LM using PEFT + TRL"

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
        build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer-caching optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create required output directories
RUN mkdir -p output lora_adapter dataset

# ─────────────────────────────────────────────────────────────────────────────
#  Default command: generate dataset → train → predict
#  You can override this in docker-compose.yml or at `docker run` time.
# ─────────────────────────────────────────────────────────────────────────────
CMD ["bash", "-c", \
     "python generate_dataset.py && \
      python train.py && \
      python predict.py"]
