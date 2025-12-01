# DNA Embedding Models - Docker Setup
# This Dockerfile ensures the project works on all computers regardless of local setup

# Use PyTorch official image with CUDA support (for GPU)
# For CPU-only, change to: pytorch/pytorch:2.1.0-cpu
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code (data files excluded via .dockerignore)
COPY src/ ./src/
COPY sweeps/ ./sweeps/
COPY evaluation/ ./evaluation/
COPY requirements.txt .
COPY README.md .

# Set Python path to include the project root
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command (can be overridden when running)
CMD ["python", "--version"]
