"""
Importer module for loading images for cell segmentation.

This module provides functions for finding and loading images from directories.
"""

import os
from typing import List, Tuple


def find_images(input_dir: str, extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')) -> List[str]:
    """
    Recursively finds all image files in a directory.

    Args:
        input_dir: Path to the root directory to search
        extensions: Tuple of file extensions to include
        
    Returns:
        List of absolute paths to image files
    """
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_relative_output_path(input_path: str, input_dir: str, output_dir: str, suffix: str = '_overlay.png') -> str:
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
    base_name, _ = os.path.splitext(relative_path)
    output_path = os.path.join(output_dir, base_name + suffix)
    return output_path