import os
import numpy as np
from typing import Tuple, Union, Any, List, Set, Optional, Dict
from tqdm import tqdm

from cellsegkit.utils.system import get_cpu_utilization, get_gpu_utilization
from cellsegkit.importer.importer import find_images
from cellsegkit.exporter.exporter import (
    save_mask_as_npy,
    save_mask_as_png,
    export_yolo_annotations,
    draw_overlay,
)

# Import augmentation components
try:
    from cellsegkit.augmentor import (
        ImageAugmentor,
        AugmentationConfig,
        create_augmentor,
    )

    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    print(
        "[WARNING] Augmentation module not found. Augmentation features will be disabled."
    )


# Valid export formats
VALID_EXPORT_FORMATS = {"overlay", "npy", "png", "yolo", "confidence"}
VALID_AGGREGATE_METHODS = {"mean", "majority", "max", "unanimous"}


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
    # New augmentation parameters
    augmentation: Optional[Union[bool, str, "ImageAugmentor"]] = None,
    augmentation_config: Optional[Dict[str, Any]] = None,
    save_augmented: bool = False,
    aggregate_method: str = "mean",
    confidence_threshold: float = 0.7,
    tta_batch_size: int = 1,  # Process augmented images in batches
) -> Dict[str, Any]:
    """
    Run full segmentation pipeline with optional augmentation support.

    Args:
        segmenter: An instance of a segmenter (must have .load_image() and .segment())
        input_dir: Directory of input images
        output_dir: Directory to save results
        export_formats: Formats to export, can be any combination of:
                       "overlay", "npy", "png", "yolo", "confidence"
                       Default is ("overlay", "npy", "png", "yolo")
        augmentation: Enable augmentation:
                     - None: No augmentation (default)
                     - True: Use default augmentation
                     - str: Use preset ("light", "default", "heavy", "geometric_only")
                     - ImageAugmentor: Use custom augmentor instance
        augmentation_config: Dict of augmentation parameters (used when augmentation=True)
        save_augmented: Whether to save augmented images (default: False)
        aggregate_method: How to combine masks from augmented images:
                         "mean", "majority", "max", "unanimous" (default: "mean")
        confidence_threshold: Threshold for confidence-based filtering (default: 0.7)
        tta_batch_size: Batch size for test-time augmentation processing

    Returns:
        Dict containing:
            - "processed": Number of successfully processed images
            - "errors": List of errors
            - "augmentation_used": Whether augmentation was used
            - "confidence_stats": Statistics about confidence if augmentation was used

    Raises:
        ValueError: If any of the specified export formats or aggregate method is invalid
    """
    # Validate export formats
    export_formats = set(export_formats)  # Convert to set for consistency
    if not export_formats:
        raise ValueError("At least one export format must be specified")

    invalid_formats = export_formats - VALID_EXPORT_FORMATS
    if invalid_formats:
        raise ValueError(
            f"Invalid export format(s): {', '.join(invalid_formats)}. "
            f"Valid formats are: {', '.join(VALID_EXPORT_FORMATS)}"
        )

    # Validate aggregate method
    if aggregate_method not in VALID_AGGREGATE_METHODS:
        raise ValueError(
            f"Invalid aggregate method: {aggregate_method}. "
            f"Valid methods are: {', '.join(VALID_AGGREGATE_METHODS)}"
        )

    # Handle augmentation parameter
    augmentor = None
    augmentation_used = False

    if augmentation is not None and AUGMENTATION_AVAILABLE:
        augmentation_used = True
        if isinstance(augmentation, bool) and augmentation:
            # Use default augmentation or custom config
            if augmentation_config:
                config = AugmentationConfig(**augmentation_config)
                augmentor = ImageAugmentor(config)
            else:
                augmentor = create_augmentor("default")
        elif isinstance(augmentation, str):
            # Use preset
            augmentor = create_augmentor(augmentation)
        elif hasattr(augmentation, "augment"):  # Duck typing for ImageAugmentor
            # Use provided augmentor
            augmentor = augmentation
        else:
            print(
                f"[WARNING] Invalid augmentation parameter: {augmentation}. Augmentation disabled."
            )
            augmentation_used = False
    elif augmentation is not None and not AUGMENTATION_AVAILABLE:
        print(
            "[WARNING] Augmentation requested but module not available. Processing without augmentation."
        )

    # Add confidence to export formats if using augmentation
    if augmentation_used and "confidence" not in export_formats:
        export_formats.add("confidence")

    # Find images
    image_paths = find_images(input_dir)
    if not image_paths:
        print(f"No images found in {input_dir}")
        return {
            "processed": 0,
            "errors": [],
            "augmentation_used": augmentation_used,
            "confidence_stats": None,
        }

    total_images = len(image_paths)
    aug_info = (
        f" with {augmentor.config.num_augmentations} augmentations each"
        if augmentation_used
        else ""
    )
    print(
        f"Found {total_images} images{aug_info}. "
        f"Exporting formats: {', '.join(sorted(export_formats))}"
    )

    # Track errors and confidence statistics
    error_files = []
    confidence_stats = (
        {"mean": [], "min": [], "below_threshold": 0} if augmentation_used else None
    )

    # Create progress bar
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
            if augmentation_used:
                postfix_dict["Aug"] = "Yes"

            pbar.set_postfix(postfix_dict)

            # Load image
            image = segmenter.load_image(image_path)

            # Get relative path for output
            relative_base = os.path.splitext(os.path.relpath(image_path, input_dir))[0]

            # Perform segmentation with or without augmentation
            if augmentor:
                # Test-time augmentation
                augmented_images = augmentor.augment(image)

                # Save augmented images if requested
                if save_augmented:
                    aug_dir = os.path.join(output_dir, "augmented", relative_base)
                    os.makedirs(aug_dir, exist_ok=True)
                    for i, aug_img in enumerate(augmented_images):
                        aug_path = os.path.join(aug_dir, f"aug_{i}.png")
                        if aug_img.dtype != np.uint8:
                            aug_img = (np.clip(aug_img, 0, 1) * 255).astype(np.uint8)
                        from PIL import Image as PILImage

                        PILImage.fromarray(aug_img).save(aug_path)

                # Segment each augmented image
                masks_list = []

                # Process in batches for memory efficiency
                for i in range(0, len(augmented_images), tta_batch_size):
                    batch = augmented_images[i : i + tta_batch_size]
                    for aug_img in batch:
                        mask = segmenter.segment(aug_img)
                        masks_list.append(mask)

                # Aggregate masks
                masks, confidence_map = aggregate_masks(
                    masks_list, method=aggregate_method, return_confidence=True
                )

                # Update confidence statistics
                if confidence_map is not None and confidence_stats is not None:
                    mean_conf = np.mean(confidence_map[confidence_map > 0])
                    min_conf = (
                        np.min(confidence_map[confidence_map > 0])
                        if np.any(confidence_map > 0)
                        else 0
                    )
                    confidence_stats["mean"].append(mean_conf)
                    confidence_stats["min"].append(min_conf)
                    if mean_conf < confidence_threshold:
                        confidence_stats["below_threshold"] += 1

            else:
                # Standard segmentation without augmentation
                masks = segmenter.segment(image)
                confidence_map = None

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

            if "confidence" in export_formats and confidence_map is not None:
                conf_path = os.path.join(
                    output_dir, "confidence", relative_base + ".npy"
                )
                os.makedirs(os.path.dirname(conf_path), exist_ok=True)
                np.save(conf_path, confidence_map.astype(np.float32))

                # Also save confidence visualization
                conf_vis_path = os.path.join(
                    output_dir, "confidence", relative_base + "_vis.png"
                )
                save_confidence_visualization(confidence_map, conf_vis_path)

            # If there were any format errors, add to the error list
            if format_errors:
                error_files.append((os.path.basename(image_path), format_errors))

        except Exception as e:
            error_files.append((os.path.basename(image_path), str(e)))
            pbar.write(f"[ERROR] Error processing {os.path.basename(image_path)}: {e}")

    pbar.close()

    # Calculate final statistics
    processed_count = total_images - len(error_files)

    # Print summary
    print(
        f"\n\n[SUCCESS] Task completed! Processed {processed_count}/{total_images} images successfully."
    )

    if augmentation_used:
        print(f"\n[INFO] Augmentation was used with method: {aggregate_method}")
        if save_augmented:
            print(
                f"[INFO] Augmented images saved in: {os.path.join(output_dir, 'augmented')}"
            )

    if confidence_stats and confidence_stats["mean"]:
        mean_overall = np.mean(confidence_stats["mean"])
        min_overall = np.min(confidence_stats["min"])
        print(f"\n[INFO] Confidence statistics:")
        print(f"  - Mean confidence: {mean_overall:.3f}")
        print(f"  - Minimum confidence: {min_overall:.3f}")
        print(
            f"  - Images below threshold ({confidence_threshold}): "
            f"{confidence_stats['below_threshold']}/{total_images}"
        )

    if error_files:
        print(f"\n[ERROR] Errors occurred in {len(error_files)} files:")
        for file_info in error_files:
            if isinstance(file_info[1], str):
                print(f"  - {file_info[0]}: {file_info[1]}")
            else:
                print(f"  - {file_info[0]}: Failed formats: {', '.join(file_info[1])}")
    else:
        print("\n[SUCCESS] No errors occurred during processing.")

    # Return summary
    return {
        "processed": processed_count,
        "errors": error_files,
        "augmentation_used": augmentation_used,
        "confidence_stats": {
            "mean": mean_overall
            if confidence_stats and confidence_stats["mean"]
            else None,
            "min": min_overall
            if confidence_stats and confidence_stats["min"]
            else None,
            "below_threshold": confidence_stats["below_threshold"]
            if confidence_stats
            else None,
        }
        if confidence_stats
        else None,
    }


def aggregate_masks(
    masks_list: List[np.ndarray], method: str = "mean", return_confidence: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Aggregate multiple segmentation masks using specified method.

    Args:
        masks_list: List of segmentation masks
        method: Aggregation method ("mean", "majority", "max", "unanimous")
        return_confidence: Whether to return confidence map

    Returns:
        If return_confidence is False: aggregated mask
        If return_confidence is True: (aggregated mask, confidence map)
    """
    if len(masks_list) == 1:
        if return_confidence:
            return masks_list[0], np.ones_like(masks_list[0], dtype=np.float32)
        return masks_list[0]

    masks_array = np.stack(masks_list, axis=0)

    if method == "mean":
        # For instance segmentation, we need to match instances across masks
        # For now, we'll use a simple averaging approach
        # In practice, you might want to use Hungarian algorithm for matching
        mean_mask = np.mean(masks_array, axis=0)
        final_mask = (mean_mask > 0.5).astype(masks_list[0].dtype)
        confidence = 1.0 - np.std(masks_array, axis=0)

    elif method == "majority":
        # Majority voting
        from scipy import stats

        final_mask = stats.mode(masks_array, axis=0)[0].squeeze()
        # Confidence based on agreement
        confidence = np.sum(masks_array == final_mask[np.newaxis, :, :], axis=0) / len(
            masks_list
        )

    elif method == "max":
        # Maximum response
        final_mask = np.max(masks_array, axis=0)
        confidence = np.mean(masks_array > 0, axis=0)

    elif method == "unanimous":
        # Only keep regions where all masks agree
        final_mask = np.min(masks_array, axis=0)
        confidence = np.ones_like(final_mask, dtype=np.float32)
        confidence[final_mask == 0] = 0

    else:
        raise ValueError(f"Unknown aggregation method: {method}")

    if return_confidence:
        return final_mask.astype(masks_list[0].dtype), confidence.astype(np.float32)
    return final_mask.astype(masks_list[0].dtype)


def save_confidence_visualization(
    confidence_map: np.ndarray, output_path: str, cmap: str = "viridis"
) -> bool:
    """
    Save confidence map as a color-coded visualization.

    Args:
        confidence_map: Confidence values (0-1)
        output_path: Path to save visualization
        cmap: Colormap name

    Returns:
        Success status
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        # Normalize to 0-255
        conf_normalized = (confidence_map * 255).astype(np.uint8)

        # Apply colormap
        colormap = cm.get_cmap(cmap)
        colored = colormap(conf_normalized)

        # Convert to RGB
        rgb_image = (colored[:, :, :3] * 255).astype(np.uint8)

        # Save
        from PIL import Image as PILImage

        PILImage.fromarray(rgb_image).save(output_path)

        return True
    except Exception as e:
        print(f"[WARNING] Could not save confidence visualization: {e}")
        return False


# Convenience functions for common use cases
def run_segmentation_with_tta(
    segmenter: Any,
    input_dir: str,
    output_dir: str,
    augmentation_preset: str = "light",
    num_augmentations: int = 3,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run segmentation with test-time augmentation using a preset.

    Args:
        segmenter: Segmenter instance
        input_dir: Input directory
        output_dir: Output directory
        augmentation_preset: Preset name ("light", "default", "heavy")
        num_augmentations: Number of augmented versions
        **kwargs: Additional arguments for run_segmentation

    Returns:
        Results dictionary from run_segmentation
    """
    # Create augmentor with specified number of augmentations
    if AUGMENTATION_AVAILABLE:
        augmentor = create_augmentor(augmentation_preset)
        augmentor.config.num_augmentations = num_augmentations
    else:
        print("[WARNING] Augmentation not available, running without TTA")
        augmentor = None

    return run_segmentation(
        segmenter=segmenter,
        input_dir=input_dir,
        output_dir=output_dir,
        augmentation=augmentor,
        **kwargs,
    )


def run_segmentation_simple(
    segmenter: Any,
    input_dir: str,
    output_dir: str,
) -> None:
    """
    Simple wrapper for backward compatibility.
    Runs segmentation without any augmentation.
    """
    run_segmentation(
        segmenter=segmenter,
        input_dir=input_dir,
        output_dir=output_dir,
        augmentation=None,
    )
