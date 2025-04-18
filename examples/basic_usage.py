"""
Basic usage example for CellSegKit.

This script demonstrates how to use CellSegKit to segment cells in images.
"""

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

print(f"Segmentation complete. Results saved to {output_dir}")