# CellSegKit

A toolkit for cell segmentation using Cellpose and CellSAM models.

## Features

- **Model Loading**: Factory pattern for loading Cellpose and CellSAM models
- **Image Importing**: Recursive importing of PNG, JPEG, TIFF images
- **Result Exporting**: Export segmentation results as:
  - YOLO format annotations
  - NumPy arrays
  - Indexed PNG masks
  - Visual overlays
- **Unified Pipeline**: Simple API for running the complete segmentation workflow
- **Format Conversion**: Convert between mask formats without re-running segmentation
- **Resource Monitoring**: Track CPU and GPU utilization during processing

## Installation

### Basic Installation

```bash
pip install git+https://github.com/nazarzharskyi/cryobiology3.git
```

### With Optional Dependencies

You can install optional dependencies by specifying extras:

```bash
# With CellSAM Support
pip install "git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[cellsam]"

# With YOLO Support
pip install "git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[yolo]"

# With GPU Monitoring
pip install "git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[gpu]"

# With Development Tools
pip install "git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[dev]"

# Full Installation (with all optional dependencies)
pip install "git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[all]"
```

### Local Installation (after cloning the repository)

```bash
# Clone the repository
git clone https://github.com/nazarzharskyi/cryobiology3.git
cd cryobiology3

# Basic installation
pip install .

# With optional dependencies
pip install ".[cellsam]"  # For CellSAM support
pip install ".[yolo]"     # For YOLO format support
pip install ".[gpu]"      # For GPU monitoring
pip install ".[dev]"      # For development tools
pip install ".[all]"      # For all optional dependencies
```

## Quick Start

```python
import os
from cellsegkit import SegmenterFactory, run_segmentation

# Define input and output directories
input_dir = "dataset"
output_dir = "output"

# Create a segmenter (Cellpose or CellSAM)
segmenter = SegmenterFactory.create(
    model_type="cyto",  # Options: "cyto", "nuclei", "cellpose", "cellsam"
    use_gpu=True,       # Use GPU if available
)

# Run segmentation on a folder of images
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=output_dir,
    export_formats=("overlay", "npy", "png", "yolo")  # Choose which formats to export
)
```

### CLI Usage (main.py)

```bash
# Basic usage
python main.py --model cyto --input dataset --output results

# Specify export formats
python main.py --model cyto --input dataset --output results --export overlay,npy

# Use CellSAM model
python main.py --model cellsam --input dataset --output results
```

### Selective Export Examples

```python
# Export only visualization formats
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir="output/visualization_only",
    export_formats=("overlay", "png")  # Only visualization formats
)

# Export only data formats
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir="output/data_only",
    export_formats=("npy", "yolo")  # Only data formats
)
```

## Configuration

### GPU Detection

CellSegKit automatically detects and utilizes GPU acceleration when available:

```python
from cellsegkit.utils.gpu_utils import get_device, check_gpu_availability

# Check if GPU is available
gpu_available = check_gpu_availability(verbose=True)
print(f"GPU available: {gpu_available}")

# Get the appropriate device (CUDA or CPU)
device = get_device(prefer_gpu=True)
print(f"Using device: {device}")
```

### JSON Configuration

You can also use JSON files for configuration:

```json
{
  "model": "cyto",
  "input_dir": "dataset",
  "output_dir": "results",
  "export_formats": ["overlay", "npy"],
  "use_gpu": true
}
```

### CLI Configuration

When using the CLI, you can configure the segmentation process with various flags:

```bash
python main.py --model cyto --input dataset --output results --export overlay,npy --gpu
```

## API Reference

### SegmenterFactory.create()

```python
SegmenterFactory.create(model_type: str, use_gpu=True, sam_checkpoint_path=None)
```

- **model_type**: Type of model to create ("cyto", "nuclei", or "cellsam")
- **use_gpu**: Whether to use GPU acceleration if available
- **sam_checkpoint_path**: Path to SAM checkpoint file (for CellSAM only)

Returns an instance of a segmenter class.

### run_segmentation()

```python
run_segmentation(
    segmenter: Any,
    input_dir: str,
    output_dir: str,
    export_formats: Union[Tuple[str, ...], List[str], Set[str]] = ("overlay", "npy", "png", "yolo")
)
```

- **segmenter**: An instance of a segmenter (must have .load_image() and .segment())
- **input_dir**: Directory of input images
- **output_dir**: Directory to save results
- **export_formats**: Formats to export, can be any combination of: "overlay", "npy", "png", "yolo"

### get_device()

```python
get_device(prefer_gpu: bool = True)
```

- **prefer_gpu**: Whether to prefer GPU over CPU if available

Returns the appropriate device (CUDA or CPU) based on availability and preference.

## Docker

### Building the Docker Image

```bash
# Build the Docker image
docker build -t cellsegkit .
```

### Running the Container

```bash
# Run with basic options
docker run -v /path/to/input:/input -v /path/to/output:/output cellsegkit --model cyto --input /input --output /output

# Run with specific export formats
docker run -v /path/to/input:/input -v /path/to/output:/output cellsegkit --model cyto --input /input --output /output --export overlay,npy
```

### Using docker-compose

Create a `docker-compose.yml` file:

```yaml
version: '3'
services:
  cellsegkit:
    build: .
    volumes:
      - ./dataset:/input
      - ./results:/output
    command: --model cyto --input /input --output /output
```

Then run:

```bash
docker-compose up
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT
