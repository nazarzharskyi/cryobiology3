"""
Pipeline module for cell segmentation.

This module provides a unified workflow for cell segmentation, combining
model loading, image importing, segmentation, and result exporting.
"""

import os
from typing import Tuple, Union, Any, List, Set

from cellsegkit.importer.importer import find_images
from cellsegkit.exporter.exporter import (
    save_mask_as_npy,
    save_mask_as_png,
    export_yolo_annotations,
    draw_overlay,
)


# Valid export formats
VALID_EXPORT_FORMATS = {"overlay", "npy", "png", "yolo"}


def run_segmentation(
    segmenter: Any,
    input_dir: str,
    output_dir: str,
    export_formats: Union[Tuple[str, ...], List[str], Set[str]] = ("overlay", "npy", "png", "yolo")
) -> None:
    """
    Run full segmentation pipeline using a given segmenter on a folder of images.

    Args:
        segmenter: An instance of a segmenter (must have .load_image() and .segment())
        input_dir: Directory of input images
        output_dir: Directory to save results
        export_formats: Formats to export, can be any combination of: "overlay", "npy", "png", "yolo"
                       Default is all formats.

    Raises:
        ValueError: If any of the specified export formats is invalid
    """
    # Validate export formats
    if not export_formats:
        raise ValueError("At least one export format must be specified")

    invalid_formats = set(export_formats) - VALID_EXPORT_FORMATS
    if invalid_formats:
        raise ValueError(
            f"Invalid export format(s): {', '.join(invalid_formats)}. "
            f"Valid formats are: {', '.join(VALID_EXPORT_FORMATS)}"
        )

    # Find images
    image_paths = find_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return

    print(f"Found {len(image_paths)} images. Exporting formats: {', '.join(export_formats)}")

    # Process each image
    for image_path in image_paths:
        try:
            # Load and segment image
            image = segmenter.load_image(image_path)
            masks = segmenter.segment(image)

            # Get relative path for output
            relative_base = os.path.splitext(os.path.relpath(image_path, input_dir))[0]

            # Export in selected formats
            if "overlay" in export_formats:
                overlay_path = os.path.join(output_dir, "overlay", relative_base + ".png")
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                draw_overlay(image, masks, overlay_path)

            if "npy" in export_formats:
                npy_path = os.path.join(output_dir, "npy", relative_base + ".npy")
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                save_mask_as_npy(masks, npy_path)

            if "png" in export_formats:
                png_path = os.path.join(output_dir, "png", relative_base + ".png")
                os.makedirs(os.path.dirname(png_path), exist_ok=True)
                save_mask_as_png(masks, png_path)

            if "yolo" in export_formats:
                txt_path = os.path.join(output_dir, "yolo", relative_base + ".txt")
                os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                image_height, image_width = image.shape[:2]
                export_yolo_annotations(masks, txt_path, (image_width, image_height))

            print(f"✅ Finished: {image_path}")

        except Exception as e:
            print(f"❌ Error processing {image_path}: {e}")
