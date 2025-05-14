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

### Mask Format Conversion

You can convert between different mask formats without re-running segmentation:

```python
from cellsegkit import convert_mask_format

# Convert a .npy mask to PNG format
convert_mask_format(
    mask_path="output/data_only/npy/image1.npy",
    output_format="png",
    output_path="output/converted/image1.png"
)

# Convert a mask to YOLO format (requires original image)
convert_mask_format(
    mask_path="output/data_only/npy/image1.npy",
    output_format="yolo",
    output_path="output/converted/image1.txt",
    original_image_path="dataset/image1.tif"
)

# Create an overlay visualization from a mask (requires original image)
convert_mask_format(
    mask_path="output/data_only/npy/image1.npy",
    output_format="overlay",
    output_path="output/converted/image1_overlay.png",
    original_image_path="dataset/image1.tif"
)
```

See the `examples/mask_conversion.py` file for a complete example of mask format conversion.

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

### convert_mask_format()

```python
convert_mask_format(
    mask_path: str,
    output_format: str,
    output_path: str,
    original_image_path: Optional[str] = None,
    class_id: int = 0
)
```

- **mask_path**: Path to the input mask file (.npy or .png)
- **output_format**: Desired output format ("npy", "png", "yolo", or "overlay")
- **output_path**: Path where the converted file will be saved
- **original_image_path**: Path to the original image (required for "overlay" and "yolo" formats)
- **class_id**: Class ID to assign to all bounding boxes for YOLO format (default: 0)

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
