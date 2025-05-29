"""
Advanced usage example for CellSegKit.

This script demonstrates how to use individual components of CellSegKit
for a custom cell segmentation workflow.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from cellsegkit.loader import CellposeSegmenter, CellSAMSegmenter
from cellsegkit.importer import find_images
from cellsegkit.exporter import (
    save_mask_as_npy,
    save_mask_as_png,
    export_yolo_annotations,
    draw_overlay,
)

# Define input and output directories
input_dir = r"C:\fun\dataset"
output_dir = r"C:\fun\output"

# Create output directories
os.makedirs(os.path.join(output_dir, "cellpose"), exist_ok=True)
os.makedirs(os.path.join(output_dir, "cellsam"), exist_ok=True)

# Find all images in the input directory
image_paths = find_images(input_dir)
print(f"Found {len(image_paths)} images")

# Create segmenters
cellpose_segmenter = CellposeSegmenter(model_type="cyto", use_gpu=False)
cellsam_segmenter = CellSAMSegmenter(use_gpu=False)

# Process all images in batch
print("Running Cellpose segmentation on all images...")
cellpose_results = cellpose_segmenter.batch_segment(image_paths)
print(f"Completed Cellpose segmentation for {len(cellpose_results)} images")

"""
print("Running CellSAM segmentation on all images...")
cellsam_results = cellsam_segmenter.batch_segment(image_paths)
print(f"Completed CellSAM segmentation for {len(cellsam_results)} images")
"""

# Process the Cellpose results
print("Exporting Cellpose results...")
for image_path, cellpose_mask in cellpose_results.items():
    # Get base filename for output
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load the original image for visualization
    cellpose_image = cellpose_segmenter.load_image(image_path)

    # Export Cellpose results
    cellpose_output_base = os.path.join(output_dir, "cellpose", base_name)

    # Save as NumPy array
    save_mask_as_npy(cellpose_mask, f"{cellpose_output_base}.npy")

    # Save as PNG
    save_mask_as_png(cellpose_mask, f"{cellpose_output_base}.png")

    # Export as YOLO annotations
    image_height, image_width = cellpose_image.shape[:2]
    export_yolo_annotations(
        cellpose_mask, f"{cellpose_output_base}.txt", (image_width, image_height)
    )

    # Create overlay visualization
    draw_overlay(cellpose_image, cellpose_mask, f"{cellpose_output_base}_overlay.png")

"""
# Process the CellSAM results
print("Exporting CellSAM results...")
for image_path, cellsam_mask in cellsam_results.items():
    # Get base filename for output
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Load the original image for visualization
    cellsam_image = cellsam_segmenter.load_image(image_path)

    # Export CellSAM results
    cellsam_output_base = os.path.join(output_dir, "cellsam", base_name)

    # Save as NumPy array
    save_mask_as_npy(cellsam_mask, f"{cellsam_output_base}.npy")

    # Save as PNG
    save_mask_as_png(cellsam_mask, f"{cellsam_output_base}.png")

    # Create overlay visualization
    # Convert grayscale to RGB for visualization if needed
    if len(cellsam_image.shape) == 2:
        cellsam_image_rgb = np.stack([cellsam_image] * 3, axis=-1)
    else:
        cellsam_image_rgb = cellsam_image

    draw_overlay(cellsam_image_rgb, cellsam_mask, f"{cellsam_output_base}_overlay.png")

print(f"Processing complete. All results saved to {output_dir}")
"""
