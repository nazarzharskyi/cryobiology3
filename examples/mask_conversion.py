"""
Mask conversion example for CellSegKit.

This script demonstrates how to convert between different mask formats
without re-running segmentation.
"""

import os
from cellsegkit import convert_mask_format

# Define input and output directories
input_dir = "output"
output_dir = "output/mask_conversion"
os.makedirs(output_dir, exist_ok=True)

# Example 1: Convert .npy mask to PNG
print("\n--- Example 1: Convert .npy mask to PNG ---")
convert_mask_format(
    mask_path=os.path.join(input_dir, "data_only/npy/image1.npy"),
    output_format="png",
    output_path=os.path.join(output_dir, "npy_to_png.png"),
)

# Example 2: Convert .npy mask to YOLO format
print("\n--- Example 2: Convert .npy mask to YOLO format ---")
convert_mask_format(
    mask_path=os.path.join(input_dir, "data_only/npy/image1.npy"),
    output_format="yolo",
    output_path=os.path.join(output_dir, "npy_to_yolo.txt"),
    original_image_path="dataset/train/image1.tif",
)

# Example 3: Convert .png mask to overlay visualization
print("\n--- Example 3: Convert .png mask to overlay visualization ---")
convert_mask_format(
    mask_path=os.path.join(input_dir, "visualization_only/png/image1.png"),
    output_format="overlay",
    output_path=os.path.join(output_dir, "png_to_overlay.png"),
    original_image_path="dataset/train/image1.tif",
)

# Example 4: Convert .png mask to .npy format
print("\n--- Example 4: Convert .png mask to .npy format ---")
convert_mask_format(
    mask_path=os.path.join(input_dir, "all_formats/png/image1.png"),
    output_format="npy",
    output_path=os.path.join(output_dir, "png_to_npy.npy"),
)

print(f"\nMask conversion complete. Results saved to {output_dir}")
