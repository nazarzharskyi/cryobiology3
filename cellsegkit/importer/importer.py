"""
Importer module for loading images for cell segmentation.

This module provides functions for finding and loading images from directories,
with support for various image formats including .png, .jpg, .jpeg, .tiff, and .lsm.
"""

import os
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from PIL import Image
import tifffile


def find_images(
    input_dir: str, 
    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tiff', '.tif', '.lsm'),
    recursive: bool = True
) -> List[str]:
    """
    Recursively finds all image files in a directory.

    Args:
        input_dir: Path to the root directory to search
        extensions: Tuple of file extensions to include
        recursive: Whether to search subdirectories recursively (default: True)
        
    Returns:
        List of absolute paths to image files

    Raises:
        FileNotFoundError: If the input directory doesn't exist
        ValueError: If the input directory is not a valid directory
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    if not os.path.isdir(input_dir):
        raise ValueError(f"Input path is not a directory: {input_dir}")
    
    image_paths = []
    
    if recursive:
        # Recursive search through all subdirectories
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(extensions):
                    image_paths.append(os.path.join(root, file))
    else:
        # Non-recursive search (only in the specified directory)
        for file in os.listdir(input_dir):
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(input_dir, file))
    
    return sorted(image_paths)


def load_image_with_metadata(
    image_path: str
) -> Tuple[np.ndarray, Dict[str, Union[Tuple[int, ...], int]]]:
    """
    Load an image and return it along with its metadata.

    Args:
        image_path: Path to the image file
        
    Returns:
        Tuple containing:
            - Image as a numpy array
            - Dictionary with metadata (shape, channels, etc.)
            
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image format is unsupported or can't be loaded
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    metadata = {}
    image = None
    
    # Handle different file formats
    if image_path.lower().endswith(('.tif', '.tiff', '.lsm')):
        try:
            # Use tifffile for TIFF and LSM formats
            with tifffile.TiffFile(image_path) as tif:
                image = tif.asarray()
                
                # Extract metadata from tags if available
                if hasattr(tif, 'pages') and len(tif.pages) > 0:
                    tags = tif.pages[0].tags
                    if 'ImageDescription' in tags:
                        metadata['description'] = tags['ImageDescription'].value
            
        except Exception as e:
            raise ValueError(f"Failed to load TIFF/LSM image: {e}")
    else:
        try:
            # Use PIL for other formats
            with Image.open(image_path) as img:
                image = np.array(img)
                metadata['mode'] = img.mode
                metadata['format'] = img.format
        except Exception as e:
            raise ValueError(f"Failed to load image: {e}")
    
    # Add basic metadata
    metadata['shape'] = image.shape
    
    # Determine number of channels
    if len(image.shape) == 2:
        # Grayscale image
        metadata['channels'] = 1
    elif len(image.shape) == 3:
        # Color image or stack
        if image.shape[2] <= 4:  # Assuming RGB or RGBA
            metadata['channels'] = image.shape[2]
        else:
            # Might be a z-stack
            metadata['channels'] = 1
            metadata['z_slices'] = image.shape[2]
    
    return image, metadata


def get_relative_output_path(
    input_path: str, 
    input_dir: str, 
    output_dir: str, 
    suffix: str = '_output'
) -> str:
    """
    Generates the output path preserving the folder structure of input_dir.

    Args:
        input_path: Path to the input image
        input_dir: Root input directory
        output_dir: Root output directory
        suffix: Suffix to add to the output file name
        
    Returns:
        Path to the output file
    """
    relative_path = os.path.relpath(input_path, input_dir)
    base_name, ext = os.path.splitext(relative_path)
    output_path = os.path.join(output_dir, base_name + suffix + ext)
    return output_path


def batch_load_images(
    image_paths: List[str],
    with_metadata: bool = False
) -> Union[List[np.ndarray], List[Tuple[np.ndarray, Dict]]]:
    """
    Load multiple images from a list of paths.

    Args:
        image_paths: List of paths to image files
        with_metadata: Whether to include metadata for each image
        
    Returns:
        If with_metadata is False: List of images as numpy arrays
        If with_metadata is True: List of tuples (image, metadata)
        
    Raises:
        FileNotFoundError: If any image file doesn't exist
    """
    results = []
    
    for path in image_paths:
        if with_metadata:
            image, metadata = load_image_with_metadata(path)
            results.append((image, metadata))
        else:
            # Just load the image without metadata
            if path.lower().endswith(('.tif', '.tiff', '.lsm')):
                image = tifffile.imread(path)
            else:
                image = np.array(Image.open(path))
            results.append(image)
    
    return results