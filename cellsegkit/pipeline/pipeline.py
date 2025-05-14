"""
Pipeline module for cell segmentation.

This module provides a unified workflow for cell segmentation, combining
model loading, image importing, segmentation, and result exporting.
"""

import os
from typing import Tuple, Union, Any, List, Set
from tqdm import tqdm

from cellsegkit.utils.system import get_cpu_utilization, get_gpu_utilization
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
    export_formats: Union[Tuple[str, ...], List[str], Set[str]] = (
        "overlay",
        "npy",
        "png",
        "yolo",
    ),
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

    total_images = len(image_paths)
    print(
        f"Found {total_images} images. Exporting formats: {', '.join(export_formats)}"
    )

    # Track errors
    error_files = []

    pbar = tqdm(
        image_paths,
        desc="Processing images",
        total=total_images,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
    )

    # Process each image
    for idx, image_path in enumerate(pbar, 1):
        try:
            # Calculate progress percentage
            cpu_util = get_cpu_utilization()
            gpu_util = get_gpu_utilization()

            postfix_dict = {
                "CPU": f"{cpu_util:.1f}%",
                "File": os.path.basename(image_path),
            }
            if gpu_util is not None:
                postfix_dict["GPU"] = f"{gpu_util:.1f}%"

            pbar.set_postfix(postfix_dict)

            # Load and segment image
            image = segmenter.load_image(image_path)
            masks = segmenter.segment(image)

            # Get relative path for output
            relative_base = os.path.splitext(os.path.relpath(image_path, input_dir))[0]

            # Track format-specific errors
            format_errors = []

            # Export in selected formats
            if "overlay" in export_formats:
                overlay_path = os.path.join(
                    output_dir, "overlay", relative_base + ".png"
                )
                os.makedirs(os.path.dirname(overlay_path), exist_ok=True)
                if not draw_overlay(image, masks, overlay_path, silent=True):
                    format_errors.append("overlay")

            if "npy" in export_formats:
                npy_path = os.path.join(output_dir, "npy", relative_base + ".npy")
                os.makedirs(os.path.dirname(npy_path), exist_ok=True)
                if not save_mask_as_npy(masks, npy_path, silent=True):
                    format_errors.append("npy")

            if "png" in export_formats:
                png_path = os.path.join(output_dir, "png", relative_base + ".png")
                os.makedirs(os.path.dirname(png_path), exist_ok=True)
                if not save_mask_as_png(masks, png_path, silent=True):
                    format_errors.append("png")

            if "yolo" in export_formats:
                txt_path = os.path.join(output_dir, "yolo", relative_base + ".txt")
                os.makedirs(os.path.dirname(txt_path), exist_ok=True)
                image_height, image_width = image.shape[:2]
                if not export_yolo_annotations(
                    masks, txt_path, (image_width, image_height), silent=True
                ):
                    format_errors.append("yolo")

            # If there were any format errors, add to the error list
            if format_errors:
                error_files.append((os.path.basename(image_path), format_errors))

        except Exception as e:
            error_files.append((os.path.basename(image_path), str(e)))
            pbar.write(f"❌ Error processing {os.path.basename(image_path)}: {e}")

    pbar.close()

    # Print summary - add a newline to move to the next line after the progress bar
    print(f"\n\n✅ Task completed! Processed {total_images} images.")

    if error_files:
        print(f"\n❌ Errors occurred in {len(error_files)} files:")
        for file_info in error_files:
            if isinstance(file_info[1], str):
                print(f"  - {file_info[0]}: {file_info[1]}")
            else:
                print(f"  - {file_info[0]}: Failed formats: {', '.join(file_info[1])}")
    else:
        print("\n✅ No errors occurred during processing.")
