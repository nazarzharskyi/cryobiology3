# Docker Setup for CellSegKit

This guide explains how to use Docker to run CellSegKit with both CPU and GPU support.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed
- For GPU support: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

## Quick Start

### Prepare Your Data

1. Create an `images` directory in your project folder and place your input images there:

```bash
mkdir -p images
# Copy your images to the images directory
```

2. Create a `results` directory where the segmentation results will be saved:

```bash
mkdir -p results
```

## Building and Running

### CPU Usage

1. Build the Docker image:

```bash
docker build -t cellseg .
```

2. Run with Docker Compose:

```bash
docker-compose up
```

### GPU Usage

1. Build the Docker image:

```bash
docker build -t cellseg .
```

2. Run with GPU support using Docker Compose:

```bash
# Edit docker-compose.yml to uncomment the GPU configuration
# Then run:
docker-compose --compatibility up
```

Alternatively, run directly with Docker:

```bash
docker run --gpus all -v ./images:/app/images -v ./results:/app/results cellseg
```

## Configuration

You can configure the container using environment variables in the `docker-compose.yml` file:

- `MODEL_TYPE`: The segmentation model to use (options: `cyto`, `nuclei`, `cellpose`, `cellsam`)
- `EXPORT_FORMATS`: Comma-separated list of export formats (options: `overlay`, `npy`, `png`, `yolo`)
- `USE_GPU`: Set to `true` to enable GPU usage, `false` to disable

Example:

```yaml
environment:
  - MODEL_TYPE=cellsam
  - EXPORT_FORMATS=overlay,png
  - USE_GPU=true
```

## Advanced Usage

### Using a Dedicated GPU Image

For optimal GPU performance, you can build a dedicated GPU image:

1. Create a `Dockerfile.gpu` file:

```dockerfile
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir pynvml>=11.0.0

# Copy the project code
COPY . .

# Install the package in development mode
RUN pip3 install -e ".[all]"

# Create directories for mounted volumes
RUN mkdir -p /app/images /app/results

# Set default environment variables
ENV MODEL_TYPE=cyto \
    EXPORT_FORMATS=overlay,npy,png,yolo \
    USE_GPU=true \
    INPUT_DIR=/app/images \
    OUTPUT_DIR=/app/results

# Set the entrypoint
ENTRYPOINT ["python3", "main.py"]
```

2. Build the GPU image:

```bash
docker build -f Dockerfile.gpu -t cellseg:gpu .
```

3. Run the GPU image:

```bash
docker run --gpus all -v ./images:/app/images -v ./results:/app/results cellseg:gpu
```

## Best Practices

### Container Size Optimization

- Use multi-stage builds for smaller images
- Remove unnecessary packages and files
- Use `.dockerignore` to exclude unnecessary files from the build context

### Security

- Run the container as a non-root user
- Use specific package versions to avoid unexpected updates
- Scan your images for vulnerabilities using tools like Docker Scout or Trivy

### Distribution

- Push your images to a container registry (Docker Hub, GitHub Container Registry, etc.)
- Tag your images with specific versions
- Document the required environment variables and volume mounts

Example of pushing to Docker Hub:

```bash
# Tag the image
docker tag cellseg:latest username/cellseg:latest

# Push to Docker Hub
docker push username/cellseg:latest
```

## Troubleshooting

### GPU Issues

- Verify NVIDIA drivers are installed: `nvidia-smi`
- Check NVIDIA Container Toolkit installation: `docker info | grep -i nvidia`
- Ensure the container has access to the GPU: `docker run --gpus all --rm nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi`

### Permission Issues

If you encounter permission issues with the mounted volumes:

```bash
# Fix permissions on the host
chmod -R 777 ./images ./results
```

### Memory Issues

If you encounter out-of-memory errors:

```bash
# Limit container memory
docker run --gpus all --memory=8g -v ./images:/app/images -v ./results:/app/results cellseg
```