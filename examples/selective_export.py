"""
Selective export example for CellSegKit.

This script demonstrates how to selectively export segmentation results
in specific formats based on your needs.
"""

import os
from cellsegkit import SegmenterFactory, run_segmentation

# Define input and output directories
input_dir = "dataset"
output_dir = "output/selective_export"

# Create a segmenter (Cellpose in this example)
segmenter = SegmenterFactory.create(
    model_type="cyto",
    use_gpu=True,
)

# Available export formats:
# - "overlay": Visual overlay of segmentation boundaries on the original image
# - "npy": NumPy array for further processing in Python
# - "png": Indexed PNG mask for use in other software
# - "yolo": YOLO format annotations for object detection tasks

# Example: Export only visualization formats
# This is useful when you only need visual results for presentations or reports
print("\nExporting only visualization formats...")
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=os.path.join(output_dir, "visualization"),
    export_formats=["overlay", "png"]  # Using a list instead of tuple (both work)
)

# Example: Export only machine learning compatible formats
# This is useful when preparing data for training object detection models
print("\nExporting only machine learning formats...")
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=os.path.join(output_dir, "ml_formats"),
    export_formats=["yolo"]  # Only YOLO format
)

# Example: Export only data formats for analysis
# This is useful when you need to perform custom analysis on the segmentation masks
print("\nExporting only data analysis formats...")
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=os.path.join(output_dir, "analysis"),
    export_formats=["npy"]  # Only NumPy arrays
)

# Example: Export combination based on specific workflow needs
# This is useful for a workflow that requires both visualization and data analysis
print("\nExporting custom combination for specific workflow...")
run_segmentation(
    segmenter=segmenter,
    input_dir=input_dir,
    output_dir=os.path.join(output_dir, "custom_workflow"),
    export_formats=["overlay", "npy"]  # Visualization + data analysis
)

print("\nSelective export examples complete!")
print("Results saved to subdirectories in:", output_dir)
print("\nAvailable export formats:")
print("- overlay: Visual boundaries on original images")
print("- png: Indexed PNG masks")
print("- npy: NumPy arrays for Python processing")
print("- yolo: YOLO format annotations for object detection")