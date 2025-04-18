"""
Exporter module for saving segmentation results.

This module provides functions for exporting segmentation masks in various formats,
including numpy arrays, PNG images, YOLO annotations, and visual overlays.
"""

import numpy as np
from PIL import Image
import os
import cv2
from typing import Tuple
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries


def save_mask_as_npy(mask: np.ndarray, output_path: str) -> None:
    """
    Saves the segmentation mask as a .npy file.

    Args:
        mask: Input mask as a numpy array
        output_path: Path where the .npy file will be saved
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, mask)
        print(f"✅ Mask saved as .npy: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save mask as .npy: {e}")


def save_mask_as_png(mask: np.ndarray, output_path: str) -> None:
    """
    Saves the segmentation mask as an indexed PNG file.

    Args:
        mask: Input mask with integer labels as a numpy array
        output_path: Path where the PNG file will be saved
    """
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        mask_img = Image.fromarray(mask.astype(np.uint8), mode="P")  # Indexed PNG
        mask_img.save(output_path)
        print(f"✅ Mask saved as PNG: {output_path}")
    except Exception as e:
        print(f"❌ Failed to save mask as PNG: {e}")


def export_yolo_annotations(
    mask: np.ndarray, 
    output_txt_path: str, 
    image_size: Tuple[int, int], 
    class_id: int = 0
) -> None:
    """
    Converts the segmentation mask into YOLO-format bounding boxes and saves to a .txt file.

    Args:
        mask: Input mask with integer labels as a numpy array
        output_txt_path: Path where the annotations .txt will be saved
        image_size: Tuple as (width, height) of the original image
        class_id: Class ID to assign to all bounding boxes. Default is 0
    """
    try:
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        width, height = image_size
        annotations = []

        # Find unique object IDs in the mask (excluding background, label 0)
        for obj_id in np.unique(mask):
            if obj_id == 0:
                continue

            # Extract coordinates for the current object
            object_coords = np.argwhere(mask == obj_id)
            y_min, x_min = object_coords.min(axis=0)
            y_max, x_max = object_coords.max(axis=0)

            # Convert to YOLO format (normalized center_x, center_y, width, height)
            center_x = ((x_min + x_max) / 2) / width
            center_y = ((y_min + y_max) / 2) / height
            bbox_width = (x_max - x_min) / width
            bbox_height = (y_max - y_min) / height

            # Write annotation: class_id, center_x, center_y, bbox_width, bbox_height
            annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}")

        # Save annotations to file
        with open(output_txt_path, 'w') as f:
            f.write("\n".join(annotations))

        print(f"✅ Annotations saved to {output_txt_path}")
    except Exception as e:
        print(f"❌ Failed to export YOLO annotations: {e}")


def draw_overlay(image: np.ndarray, mask: np.ndarray, output_path: str) -> None:
    """
    Draw boundaries on top of the image and save as an overlay PNG.

    Args:
        image: Original image (grayscale or RGB) as numpy array
        mask: Segmentation mask (labels)
        output_path: Path to save overlay result
    """
    boundaries = find_boundaries(mask, mode='outer')

    # Handle grayscale → RGB conversion
    if len(image.shape) == 2:
        overlaid = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 1:
        overlaid = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        overlaid = image.copy()

    # Apply red boundaries
    overlaid[boundaries] = [255, 0, 0]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(overlaid)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"✅ Overlay saved: {output_path}")