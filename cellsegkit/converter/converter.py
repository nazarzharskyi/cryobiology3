"""
Converter module for CellSegKit.

This module provides functions for converting between different mask formats
without re-running segmentation. Supported formats include NumPy (.npy),
indexed PNG, YOLO annotations, COCO JSON, and visual overlays.
"""

import os
import json
import numpy as np
from PIL import Image
import cv2
from typing import Tuple, Optional, Union, List, Dict, Any
import datetime
import uuid
import shutil

from cellsegkit.utils.system import get_cpu_utilization, get_gpu_utilization
from cellsegkit.exporter.exporter import (
    save_mask_as_npy,
    save_mask_as_png,
    export_yolo_annotations,
    draw_overlay,
)


# Valid export formats
VALID_FORMATS = {"overlay", "npy", "png", "yolo", "coco"}


def _print_progress(progress_pct: float, message: Optional[str] = None) -> None:
    """
    Print progress with resource utilization information.

    Args:
        progress_pct: Progress percentage (0-100)
        message: Optional message to display
    """
    # Get resource utilization
    cpu_util = get_cpu_utilization()
    gpu_util = get_gpu_utilization()

    # Calculate number of lines that will be printed
    lines_count = 3 if gpu_util is None else 4
    if message:
        lines_count += 1

    # Clear all previous lines
    print("\033[J", end="")  # Clear from cursor to end of screen

    # Create progress bar (50 chars wide)
    bar_width = 50
    filled_width = int(progress_pct / 100 * bar_width)
    bar = '█' * filled_width + '░' * (bar_width - filled_width)

    # Print progress bar
    print(f"Progress: [{bar}] {progress_pct:.1f}%")

    # Print resource utilization bars
    cpu_bar_width = int(cpu_util / 100 * bar_width)
    cpu_bar = '█' * cpu_bar_width + '░' * (bar_width - cpu_bar_width)
    print(f"CPU Load: [{cpu_bar}] {cpu_util:.1f}%")

    if gpu_util is not None:
        gpu_bar_width = int(gpu_util / 100 * bar_width)
        gpu_bar = '█' * gpu_bar_width + '░' * (bar_width - gpu_bar_width)
        print(f"GPU Load: [{gpu_bar}] {gpu_util:.1f}%")

    # Print current file being processed (if message provided)
    if message:
        print(f"Current: {message}")

    # Move cursor back up to overwrite these lines next time
    print(f"\033[{lines_count}A", end="")


def _load_original_image(image_path: str) -> np.ndarray:
    """
    Load an original image for overlay or YOLO export.

    Args:
        image_path: Path to the original image

    Returns:
        Original image as a numpy array

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image can't be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Original image not found: {image_path}")

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Failed to load original image: {image_path}")

    return original_image


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


def load_mask_from_coco(file_path: str, image_id: Optional[int] = None, image_path: Optional[str] = None) -> np.ndarray:
    """
    Load a segmentation mask from a COCO JSON file.

    Args:
        file_path: Path to the COCO JSON file
        image_id: ID of the image to extract mask for (required if multiple images in the file)
        image_path: Path to the image file (used to determine dimensions if not in COCO file)

    Returns:
        Segmentation mask as a numpy array

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be loaded as a mask or if image_id is required but not provided
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"COCO file not found: {file_path}")

    try:
        with open(file_path, 'r') as f:
            coco_data = json.load(f)
        
        # Check if we have images in the COCO file
        if 'images' not in coco_data or not coco_data['images']:
            raise ValueError("No images found in COCO file")
        
        # If multiple images and no image_id specified, raise error
        if len(coco_data['images']) > 1 and image_id is None:
            raise ValueError("Multiple images found in COCO file. Please specify image_id.")
        
        # Get the target image
        target_image = None
        if image_id is not None:
            # Find image by ID
            for img in coco_data['images']:
                if img['id'] == image_id:
                    target_image = img
                    break
            if target_image is None:
                raise ValueError(f"Image with ID {image_id} not found in COCO file")
        else:
            # Use the first (and only) image
            target_image = coco_data['images'][0]
        
        # Get image dimensions
        width = target_image.get('width')
        height = target_image.get('height')
        
        # If dimensions not in COCO, try to get from image file
        if (width is None or height is None) and image_path:
            if os.path.exists(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    height, width = img.shape[:2]
                else:
                    raise ValueError(f"Failed to load image to determine dimensions: {image_path}")
            else:
                raise FileNotFoundError(f"Image file not found: {image_path}")
        
        if width is None or height is None:
            raise ValueError("Image dimensions not found in COCO file and no image path provided")
        
        # Create empty mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Find annotations for this image
        image_id = target_image['id']
        instance_id = 1  # Start with 1 (0 is background)
        
        for annotation in coco_data.get('annotations', []):
            if annotation.get('image_id') == image_id:
                # Get segmentation data
                segmentation = annotation.get('segmentation', [])
                
                # Process each polygon
                for polygon in segmentation:
                    # Convert flat list to points
                    points = []
                    for i in range(0, len(polygon), 2):
                        if i+1 < len(polygon):
                            points.append([polygon[i], polygon[i+1]])
                    
                    # Convert to numpy array
                    pts = np.array(points, dtype=np.int32)
                    
                    # Fill polygon with instance ID
                    cv2.fillPoly(mask, [pts], instance_id)
                    instance_id += 1
        
        return mask
    
    except Exception as e:
        raise ValueError(f"Failed to load mask from COCO file: {e}")


def load_mask_from_yolo(file_path: str, image_width: int, image_height: int) -> np.ndarray:
    """
    Load a segmentation mask from a YOLO annotation file.

    Args:
        file_path: Path to the YOLO annotation file (.txt)
        image_width: Width of the original image
        image_height: Height of the original image

    Returns:
        Segmentation mask as a numpy array

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file can't be loaded as a mask
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YOLO annotation file not found: {file_path}")

    try:
        # Create empty mask
        mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        # Read YOLO annotations
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        instance_id = 1  # Start with 1 (0 is background)
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:  # class_id, center_x, center_y, width, height
                # Parse YOLO format (normalized coordinates)
                center_x = float(parts[1]) * image_width
                center_y = float(parts[2]) * image_height
                bbox_width = float(parts[3]) * image_width
                bbox_height = float(parts[4]) * image_height
                
                # Calculate box coordinates
                x_min = int(center_x - bbox_width / 2)
                y_min = int(center_y - bbox_height / 2)
                x_max = int(center_x + bbox_width / 2)
                y_max = int(center_y + bbox_height / 2)
                
                # Ensure coordinates are within image bounds
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(image_width - 1, x_max)
                y_max = min(image_height - 1, y_max)
                
                # Fill rectangle with instance ID
                mask[y_min:y_max, x_min:x_max] = instance_id
                instance_id += 1
        
        return mask
    
    except Exception as e:
        raise ValueError(f"Failed to load mask from YOLO file: {e}")


def export_mask_to_coco(
    mask: np.ndarray, 
    output_path: str, 
    image_path: Optional[str] = None,
    image_id: int = 1,
    dataset_name: str = "CellSegKit Export",
    silent: bool = False
) -> bool:
    """
    Export a segmentation mask to COCO JSON format.

    Args:
        mask: Segmentation mask as a numpy array
        output_path: Path to save the COCO JSON file
        image_path: Path to the original image (optional)
        image_id: ID to assign to the image in COCO format
        dataset_name: Name of the dataset
        silent: If True, suppresses success messages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        height, width = mask.shape
        
        # Create COCO structure
        coco_data = {
            "info": {
                "description": dataset_name,
                "url": "",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "CellSegKit",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "cell",
                    "supercategory": "cell"
                }
            ],
            "images": [
                {
                    "id": image_id,
                    "license": 1,
                    "file_name": os.path.basename(image_path) if image_path else f"image_{image_id}.png",
                    "height": height,
                    "width": width,
                    "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            ],
            "annotations": []
        }
        
        # Find unique object IDs in the mask (excluding background, label 0)
        annotation_id = 1
        for obj_id in np.unique(mask):
            if obj_id == 0:
                continue
                
            # Create binary mask for this object
            binary_mask = (mask == obj_id).astype(np.uint8)
            
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Skip if no contours found
            if not contours:
                continue
                
            # Get area
            area = int(np.sum(binary_mask))
            
            # Get bounding box
            y_indices, x_indices = np.where(binary_mask > 0)
            if len(y_indices) == 0 or len(x_indices) == 0:
                continue
                
            x_min, y_min = int(np.min(x_indices)), int(np.min(y_indices))
            x_max, y_max = int(np.max(x_indices)), int(np.max(y_indices))
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            
            # Create segmentation (polygons)
            segmentation = []
            for contour in contours:
                # Simplify contour to reduce points
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) < 3:  # Need at least 3 points for a polygon
                    continue
                    
                # Flatten polygon points to COCO format [x1,y1,x2,y2,...]
                flattened = approx.flatten().tolist()
                if len(flattened) % 2 != 0:  # Ensure even number of coordinates
                    flattened = flattened[:-1]
                    
                segmentation.append(flattened)
            
            # Skip if no valid polygons
            if not segmentation:
                continue
                
            # Add annotation
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # Always use category 1 (cell)
                "segmentation": segmentation,
                "area": area,
                "bbox": [x_min, y_min, bbox_width, bbox_height],
                "iscrowd": 0
            })
            
            annotation_id += 1
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save COCO JSON
        with open(output_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        if not silent:
            print(f"✅ Mask exported to COCO format: {output_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to export mask to COCO format: {e}")
        return False


def combine_coco_files(input_files: List[str], output_path: str, silent: bool = False) -> bool:
    """
    Combine multiple COCO JSON files into a single file.

    Args:
        input_files: List of paths to COCO JSON files
        output_path: Path to save the combined COCO JSON file
        silent: If True, suppresses success messages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize combined COCO structure
        combined_coco = {
            "info": {
                "description": "Combined CellSegKit Export",
                "url": "",
                "version": "1.0",
                "year": datetime.datetime.now().year,
                "contributor": "CellSegKit",
                "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "cell",
                    "supercategory": "cell"
                }
            ],
            "images": [],
            "annotations": []
        }
        
        # Track highest IDs to avoid duplicates
        max_image_id = 0
        max_annotation_id = 0
        
        # Process each input file
        for file_path in input_files:
            if not os.path.exists(file_path):
                print(f"⚠️ Warning: COCO file not found, skipping: {file_path}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    coco_data = json.load(f)
                    
                # Update image IDs and add to combined file
                for image in coco_data.get("images", []):
                    old_image_id = image["id"]
                    new_image_id = max_image_id + 1
                    
                    # Update image ID
                    image["id"] = new_image_id
                    combined_coco["images"].append(image)
                    
                    # Update annotation image IDs
                    for annotation in coco_data.get("annotations", []):
                        if annotation["image_id"] == old_image_id:
                            # Update image reference
                            annotation["image_id"] = new_image_id
                            
                            # Update annotation ID
                            annotation["id"] = max_annotation_id + 1
                            max_annotation_id += 1
                            
                            # Add to combined file
                            combined_coco["annotations"].append(annotation)
                    
                    max_image_id = new_image_id
                    
            except Exception as e:
                print(f"⚠️ Warning: Failed to process COCO file {file_path}: {e}")
                continue
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save combined COCO JSON
        with open(output_path, 'w') as f:
            json.dump(combined_coco, f, indent=2)
            
        if not silent:
            print(f"✅ Combined {len(input_files)} COCO files into: {output_path}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to combine COCO files: {e}")
        return False


def split_coco_file(input_file: str, output_dir: str, silent: bool = False) -> bool:
    """
    Split a COCO JSON file with multiple images into separate files, one per image.

    Args:
        input_file: Path to the input COCO JSON file
        output_dir: Directory to save the split COCO JSON files
        silent: If True, suppresses success messages

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"COCO file not found: {input_file}")
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load input COCO file
        with open(input_file, 'r') as f:
            coco_data = json.load(f)
            
        # Check if we have images
        if 'images' not in coco_data or not coco_data['images']:
            raise ValueError("No images found in COCO file")
            
        # If only one image, just copy the file
        if len(coco_data['images']) == 1:
            output_path = os.path.join(output_dir, f"image_{coco_data['images'][0]['id']}.json")
            shutil.copy(input_file, output_path)
            
            if not silent:
                print(f"✅ COCO file contains only one image, copied to: {output_path}")
                
            return True
            
        # Process each image
        for image in coco_data['images']:
            image_id = image['id']
            
            # Create new COCO structure for this image
            single_coco = {
                "info": coco_data.get("info", {
                    "description": "CellSegKit Export",
                    "date_created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }),
                "licenses": coco_data.get("licenses", []),
                "categories": coco_data.get("categories", []),
                "images": [image],
                "annotations": []
            }
            
            # Find annotations for this image
            for annotation in coco_data.get("annotations", []):
                if annotation.get("image_id") == image_id:
                    single_coco["annotations"].append(annotation)
            
            # Generate output filename
            file_name = image.get("file_name", f"image_{image_id}")
            base_name = os.path.splitext(file_name)[0]
            output_path = os.path.join(output_dir, f"{base_name}.json")
            
            # Save single image COCO file
            with open(output_path, 'w') as f:
                json.dump(single_coco, f, indent=2)
                
            if not silent:
                print(f"✅ Extracted image {image_id} to: {output_path}")
        
        if not silent:
            print(f"✅ Split COCO file into {len(coco_data['images'])} separate files in: {output_dir}")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to split COCO file: {e}")
        return False


def convert_mask_format(
    mask_path: str,
    output_format: str,
    output_path: str,
    original_image_path: Optional[str] = None,
    class_id: int = 0,
    image_id: Optional[int] = None,
    dataset_name: str = "CellSegKit Export"
) -> None:
    """
    Convert a mask file from one format to another.

    Args:
        mask_path: Path to the input mask file (.npy, .png, .txt for YOLO, or .json for COCO)
        output_format: Desired output format ("npy", "png", "yolo", "coco", or "overlay")
        output_path: Path where the converted file will be saved
        original_image_path: Path to the original image (required for "overlay", "yolo", and some conversions)
        class_id: Class ID to assign to all bounding boxes for YOLO format. Default is 0
        image_id: ID to assign to the image in COCO format. Default is None (auto-assigned)
        dataset_name: Name of the dataset for COCO format. Default is "CellSegKit Export"

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

    # Display initial progress
    _print_progress(0.0, f"Converting mask from {os.path.basename(mask_path)} to {output_format} format")

    success = False
    error_message = None

    try:
        # Load the mask based on file extension
        if mask_path.lower().endswith(".npy"):
            mask = load_mask_from_npy(mask_path)
        elif mask_path.lower().endswith((".png", ".jpg", ".jpeg")):
            mask = load_mask_from_png(mask_path)
        elif mask_path.lower().endswith(".json"):
            # COCO format
            if not image_id and not original_image_path:
                raise ValueError("For COCO input, either image_id or original_image_path must be provided")
            mask = load_mask_from_coco(mask_path, image_id, original_image_path)
        elif mask_path.lower().endswith(".txt"):
            # YOLO format
            if not original_image_path:
                raise ValueError("Original image path is required for YOLO input format")
            # Load original image to get dimensions
            original_image = _load_original_image(original_image_path)
            image_height, image_width = original_image.shape[:2]
            mask = load_mask_from_yolo(mask_path, image_width, image_height)
        else:
            raise ValueError(f"Unsupported mask file format: {os.path.splitext(mask_path)[1]}. Supported formats are .npy, .png, .json (COCO), and .txt (YOLO)")

        # Show 50% progress after loading the mask
        _print_progress(50.0)

        # Convert to the desired format
        if output_format == "npy":
            success = save_mask_as_npy(mask, output_path, silent=True)
        elif output_format == "png":
            success = save_mask_as_png(mask, output_path, silent=True)
        elif output_format == "yolo":
            # Load original image to get dimensions if not already loaded
            if 'original_image' not in locals():
                original_image = _load_original_image(original_image_path)
            image_height, image_width = original_image.shape[:2]
            success = export_yolo_annotations(mask, output_path, (image_width, image_height), class_id, silent=True)
        elif output_format == "coco":
            success = export_mask_to_coco(
                mask, 
                output_path, 
                original_image_path, 
                image_id if image_id is not None else 1,
                dataset_name,
                silent=True
            )
        elif output_format == "overlay":
            # Load original image if not already loaded
            if 'original_image' not in locals():
                original_image = _load_original_image(original_image_path)

            # Convert BGR to RGB (OpenCV loads as BGR)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

            success = draw_overlay(original_image, mask, output_path, silent=True)

    except Exception as e:
        error_message = str(e)
        success = False

    # Print final status
    _print_progress(100.0)

    if success:
        print(f"\n\n\n\n\n✅ Task completed! Converted {os.path.basename(mask_path)} to {output_format} format: {os.path.basename(output_path)}")
    else:
        print(f"\n\n\n\n\n❌ Error converting {os.path.basename(mask_path)} to {output_format} format:")
        if error_message:
            print(f"  - {error_message}")