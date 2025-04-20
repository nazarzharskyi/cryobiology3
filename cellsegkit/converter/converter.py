"""
Converter module for CellSegKit.

This module provides functions for converting between different mask formats
without re-running segmentation.
"""

import os
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Union, Literal

from cellsegkit.utils.system import get_cpu_utilization, get_gpu_utilization
from cellsegkit.exporter.exporter import (
    save_mask_as_npy,
    save_mask_as_png,
    export_yolo_annotations,
    draw_overlay,
)


# Valid export formats
VALID_FORMATS = {"overlay", "npy", "png", "yolo"}


def load_mask_from_npy(file_path: str) -> np.ndarray:
    """
    Load a segmentation mask from a .npy file.

    Args:
        file_path: Path to the .npy file

    Returns:
        Segmentation mask as a numpy array

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be loaded as a mask
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mask file not found: {file_path}")

    try:
        mask = np.load(file_path)
        return mask
    except Exception as e:
        raise ValueError(f"Failed to load mask from .npy file: {e}")


def load_mask_from_png(file_path: str) -> np.ndarray:
    """
    Load a segmentation mask from a PNG file.

    Args:
        file_path: Path to the PNG file

    Returns:
        Segmentation mask as a numpy array

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be loaded as a mask
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Mask file not found: {file_path}")

    try:
        mask_img = Image.open(file_path)
        mask = np.array(mask_img)
        return mask
    except Exception as e:
        raise ValueError(f"Failed to load mask from PNG file: {e}")


def convert_mask_format(
    mask_path: str,
    output_format: str,
    output_path: str,
    original_image_path: Optional[str] = None,
    class_id: int = 0
) -> None:
    """
    Convert a mask file from one format to another.

    Args:
        mask_path: Path to the input mask file (.npy or .png)
        output_format: Desired output format ("npy", "png", "yolo", or "overlay")
        output_path: Path where the converted file will be saved
        original_image_path: Path to the original image (required for "overlay" and "yolo" formats)
        class_id: Class ID to assign to all bounding boxes for YOLO format. Default is 0

    Raises:
        ValueError: If the output format is invalid or if original_image_path is required but not provided
        FileNotFoundError: If the input mask file or original image file doesn't exist
    """
    # Validate output format
    if output_format not in VALID_FORMATS:
        raise ValueError(f"Invalid output format: {output_format}. Valid formats are: {', '.join(VALID_FORMATS)}")

    # Check if original image is required but not provided
    if output_format in ["overlay", "yolo"] and not original_image_path:
        raise ValueError(f"Original image path is required for '{output_format}' format")

    # Get initial resource utilization
    cpu_util = get_cpu_utilization()
    gpu_util = get_gpu_utilization()

    # Display resource utilization
    resource_info = f"CPU: {cpu_util:.1f}%"
    if gpu_util is not None:
        resource_info += f", GPU: {gpu_util:.1f}%"

    print(f"[0.0%] Converting mask from {os.path.basename(mask_path)} to {output_format} format ({resource_info})")

    success = False
    error_message = None

    try:
        # Load the mask based on file extension
        if mask_path.lower().endswith(".npy"):
            mask = load_mask_from_npy(mask_path)
        elif mask_path.lower().endswith((".png", ".jpg", ".jpeg")):
            mask = load_mask_from_png(mask_path)
        else:
            raise ValueError(f"Unsupported mask file format: {os.path.splitext(mask_path)[1]}. Supported formats are .npy and .png")

        # Show 50% progress after loading the mask
        cpu_util = get_cpu_utilization()
        gpu_util = get_gpu_utilization()
        resource_info = f"CPU: {cpu_util:.1f}%"
        if gpu_util is not None:
            resource_info += f", GPU: {gpu_util:.1f}%"
        print(f"[50.0%] ({resource_info})")

        # Convert to the desired format
        if output_format == "npy":
            success = save_mask_as_npy(mask, output_path, silent=True)
        elif output_format == "png":
            success = save_mask_as_png(mask, output_path, silent=True)
        elif output_format == "yolo":
            # Load original image to get dimensions
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Original image not found: {original_image_path}")

            original_image = cv2.imread(original_image_path)
            if original_image is None:
                raise ValueError(f"Failed to load original image: {original_image_path}")

            image_height, image_width = original_image.shape[:2]
            success = export_yolo_annotations(mask, output_path, (image_width, image_height), class_id, silent=True)
        elif output_format == "overlay":
            # Load original image
            if not os.path.exists(original_image_path):
                raise FileNotFoundError(f"Original image not found: {original_image_path}")

            original_image = cv2.imread(original_image_path)
            if original_image is None:
                raise ValueError(f"Failed to load original image: {original_image_path}")

            # Convert BGR to RGB (OpenCV loads as BGR)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            success = draw_overlay(original_image, mask, output_path, silent=True)

    except Exception as e:
        error_message = str(e)
        success = False

    # Get final resource utilization
    cpu_util = get_cpu_utilization()
    gpu_util = get_gpu_utilization()

    # Display resource utilization
    resource_info = f"CPU: {cpu_util:.1f}%"
    if gpu_util is not None:
        resource_info += f", GPU: {gpu_util:.1f}%"

    # Print final status
    print(f"[100.0%] ({resource_info})")

    if success:
        print(f"\n✅ Task completed! Converted {os.path.basename(mask_path)} to {output_format} format: {os.path.basename(output_path)}")
    else:
        print(f"\n❌ Error converting {os.path.basename(mask_path)} to {output_format} format:")
        if error_message:
            print(f"  - {error_message}")
