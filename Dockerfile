# Base image: Python 3.10 slim
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    # Install additional dependencies for GPU support
    pip install --no-cache-dir pynvml>=11.0.0

# Copy the project code
COPY . .

# Install the package in development mode
RUN pip install -e ".[all]"

# Create directories for mounted volumes
RUN mkdir -p /app/images /app/results

# Set default environment variables
ENV MODEL_TYPE=cyto \
    EXPORT_FORMATS=overlay,npy,png,yolo \
    USE_GPU=true \
    INPUT_DIR=/app/images \
    OUTPUT_DIR=/app/results

# Set the entrypoint
ENTRYPOINT ["python", "main.py"]

# Note on GPU support:
# This image can be used with NVIDIA GPUs by using the --gpus flag with Docker:
# docker run --gpus all -v /path/to/images:/app/images -v /path/to/results:/app/results cellseg
#
# For a dedicated GPU image, you can build with:
# FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
# and install Python and dependencies as above.