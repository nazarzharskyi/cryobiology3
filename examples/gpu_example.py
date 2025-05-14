"""
Example script demonstrating GPU utilities in cellsegkit.

This script shows how to:
1. Check GPU availability
2. Get detailed GPU information
3. Use the appropriate device (GPU or CPU) for segmentation
"""

import os
import numpy as np
from cellsegkit.utils.gpu_utils import get_device, check_gpu_availability, get_gpu_info
from cellsegkit.loader.model_loader import SegmenterFactory


def main():
    # Print header
    print("\n" + "=" * 50)
    print("CELLSEGKIT GPU UTILITIES EXAMPLE")
    print("=" * 50 + "\n")

    # Check GPU availability (this will print a friendly message)
    gpu_available = check_gpu_availability(verbose=True)
    print(f"\nGPU available: {gpu_available}\n")

    # Get detailed GPU information
    print("Detailed GPU Information:")
    print("-" * 30)
    gpu_info = get_gpu_info()
    for key, value in gpu_info.items():
        if isinstance(value, list) and key == "memory_info":
            print(f"{key}:")
            for device_info in value:
                print(
                    f"  - Device {device_info['device']}: {device_info['name']} ({device_info['total_memory_gb']} GB)"
                )
        else:
            print(f"{key}: {value}")
    print("-" * 30 + "\n")

    # Get the appropriate device based on availability
    device = get_device(prefer_gpu=True)
    print(f"Using device: {device}\n")

    # Example: Create a segmenter with automatic GPU/CPU selection
    print("Creating segmenters with automatic GPU/CPU selection:")
    print("-" * 30)

    # Try with GPU preference
    print("1. With GPU preference (use_gpu=True):")
    segmenter_gpu = SegmenterFactory.create("cyto", use_gpu=True)
    print(f"   - CellposeSegmenter using GPU: {segmenter_gpu.use_gpu}")

    # Try with CPU preference
    print("\n2. With CPU preference (use_gpu=False):")
    segmenter_cpu = SegmenterFactory.create("cyto", use_gpu=False)
    print(f"   - CellposeSegmenter using GPU: {segmenter_cpu.use_gpu}")

    print("-" * 30 + "\n")

    # Try to load and segment a sample image if available
    sample_image_path = os.path.join(os.path.dirname(__file__), "sample_image.jpg")
    if os.path.exists(sample_image_path):
        print(f"Found sample image: {sample_image_path}")
        print("Segmenting image...")

        # Load the image
        image = segmenter_gpu.load_image(sample_image_path)

        # Segment the image
        mask = segmenter_gpu.segment(image)

        print(f"Segmentation complete. Mask shape: {mask.shape}")
    else:
        print("No sample image found. Skipping segmentation test.")

    print("\n" + "=" * 50)
    print("EXAMPLE COMPLETE")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
