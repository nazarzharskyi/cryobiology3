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

## Installation

### Basic Installation

```bash
pip install cellsegkit
```

### With CellSAM Support

```bash
pip install "cellsegkit[cellsam]"
```

### With YOLO Support

```bash
pip install "cellsegkit[yolo]"
```

### Full Installation (with all dependencies)

```bash
pip install "cellsegkit[cellsam,yolo]"
```

## Quick Start

```python
from cellsegkit import SegmenterFactory, run_segmentation

# Create a segmenter (Cellpose or CellSAM)
segmenter = SegmenterFactory.create(
    model_type="cyto",  # Options: "cyto", "nuclei", "cellpose", "cellsam"
    use_gpu=True,       # Use GPU if available
)

# Run segmentation on a folder of images
run_segmentation(
    segmenter=segmenter,
    input_dir="path/to/input/images",
    output_dir="path/to/output/directory",
    export_formats=("overlay", "npy", "png", "yolo")  # Choose which formats to export
)
```

## Advanced Usage

### Using Cellpose Segmenter Directly

```python
from cellsegkit.loader import CellposeSegmenter

# Create a Cellpose segmenter
segmenter = CellposeSegmenter(model_type="cyto", use_gpu=True)

# Load and segment a single image
image = segmenter.load_image("path/to/image.png")
mask = segmenter.segment(image)

# Now you can use the mask with the exporter functions
```

### Using CellSAM Segmenter

```python
from cellsegkit.loader import CellSAMSegmenter
from cellsegkit.exporter import draw_overlay
import matplotlib.pyplot as plt

# Create a CellSAM segmenter
segmenter = CellSAMSegmenter(use_gpu=True)

# Load and segment a single image
image = segmenter.load_image("path/to/image.png")
mask = segmenter.segment(image)

# Export as overlay
draw_overlay(image, mask, "path/to/output/overlay.png")
```

### Custom Export Workflow

```python
from cellsegkit import SegmenterFactory
from cellsegkit.importer import find_images
from cellsegkit.exporter import save_mask_as_npy, export_yolo_annotations
import os

# Create segmenter
segmenter = SegmenterFactory.create("cyto")

# Find images
image_paths = find_images("path/to/images")

# Process each image
for image_path in image_paths:
    # Load and segment
    image = segmenter.load_image(image_path)
    mask = segmenter.segment(image)

    # Custom export logic
    base_name = os.path.splitext(os.path.basename(image_path))[0]

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

Optional dependencies:
- segment-anything (for CellSAM support)
- ultralytics (for YOLO format support)

## License

MIT
