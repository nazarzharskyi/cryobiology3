"""
Basic usage example for CellSegKit.

This script demonstrates how to use CellSegKit to segment cells in images
and selectively export the results in specific formats.
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

# Example 1: Export all formats (default behavior)
print("\n--- Example 1: Export all formats ---")
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=os.path.join(output_dir, "all_formats"),
    export_formats=("overlay", "npy", "png", "yolo")  # All formats
)

# Example 2: Export only visualization formats
print("\n--- Example 2: Export only visualization formats ---")
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=os.path.join(output_dir, "visualization_only"),
    export_formats=("overlay", "png")  # Only visualization formats
)

# Example 3: Export only data formats for further processing
print("\n--- Example 3: Export only data formats ---")
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=os.path.join(output_dir, "data_only"),
    export_formats=("npy", "yolo")  # Only data formats
)

# Example 4: Export only a single format
print("\n--- Example 4: Export only overlay visualizations ---")
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=os.path.join(output_dir, "overlay_only"),
    export_formats=("overlay",)  # Note the comma to make it a tuple with one element
)

print(f"\nSegmentation complete. Results saved to different subdirectories in {output_dir}")
