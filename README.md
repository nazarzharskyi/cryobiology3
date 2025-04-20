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

### With CellSAM Support

```bash
pip install "cellsegkit[cellsam]"
```

### With YOLO Support

```bash
pip install "cellsegkit[yolo]"
```

### With GPU Monitoring

```bash
pip install "cellsegkit[gpu]"
```

### Full Installation (with all dependencies)

```bash
pip install git+https://github.com/nazarzharskyi/cryobiology3.git
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

## Advanced Usage

### Using Cellpose Segmenter Directly

```python
from cellsegkit.loader import CellposeSegmenter
from cellsegkit.exporter import save_mask_as_npy

# Create a Cellpose segmenter
segmenter = CellposeSegmenter(model_type="cyto", use_gpu=True)

# Load and segment a single image
image = segmenter.load_image("path/to/image.png")
mask = segmenter.segment(image)

# Save the mask as a NumPy array
save_mask_as_npy(mask, "path/to/output/mask.npy")
```

### Using CellSAM Segmenter

```python
from cellsegkit.loader import CellSAMSegmenter
from cellsegkit.exporter import draw_overlay

# Create a CellSAM segmenter
segmenter = CellSAMSegmenter(use_gpu=True)

# Load and segment a single image
image = segmenter.load_image("path/to/image.png")
mask = segmenter.segment(image)

# Export as overlay
draw_overlay(image, mask, "path/to/output/overlay.png")
```

### Converting Between Mask Formats

```python
import os
from cellsegkit import convert_mask_format

# Define input and output directories
input_dir = "output"
output_dir = "output/mask_conversion"
os.makedirs(output_dir, exist_ok=True)

# Convert from .npy to .png
convert_mask_format(
    mask_path=os.path.join(input_dir, "data_only/npy/train/image1.npy"),
    output_format="png",
    output_path=os.path.join(output_dir, "npy_to_png.png")
)

# Convert from .npy to YOLO format (requires original image)
convert_mask_format(
    mask_path=os.path.join(input_dir, "data_only/npy/train/image1.npy"),
    output_format="yolo",
    output_path=os.path.join(output_dir, "npy_to_yolo.txt"),
    original_image_path="dataset/train/image1.tif"  # Required for YOLO format
)
```

### Custom Export Workflow

```python
import os
from cellsegkit import SegmenterFactory
from cellsegkit.importer import find_images
from cellsegkit.exporter import save_mask_as_npy, export_yolo_annotations

# Create segmenter
segmenter = SegmenterFactory.create("cyto")

# Find images
image_paths = find_images("dataset")

# Process each image
for image_path in image_paths:
    # Load and segment
    image = segmenter.load_image(image_path)
    mask = segmenter.segment(image)

    # Custom export logic
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Create output directories
    os.makedirs("output/npy", exist_ok=True)
    os.makedirs("output/yolo", exist_ok=True)

    # Save as NumPy array
    save_mask_as_npy(mask, f"output/npy/{base_name}.npy")

    # Export YOLO annotations
    image_height, image_width = image.shape[:2]
    export_yolo_annotations(
        mask, 
        f"output/yolo/{base_name}.txt", 
        (image_width, image_height)
    )
```

## Project Structure

```
cellsegkit/
├── __init__.py           # Main package exports
├── converter/            # Mask format conversion
├── exporter/             # Export segmentation results
├── importer/             # Import images
├── loader/               # Load segmentation models
├── pipeline/             # Unified segmentation workflow
└── utils/                # Utility functions
```

## Requirements

- Python 3.8+
- cellpose
- numpy
- opencv-python
- pillow
- matplotlib
- scikit-image
- torch
- torchvision
- psutil

Optional dependencies:
- segment-anything, cellSAM (for CellSAM support)
- ultralytics (for YOLO format support)
- pynvml (for GPU monitoring)

## License

MIT
