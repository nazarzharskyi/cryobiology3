"""
Mask conversion example for CellSegKit.

This script demonstrates how to convert between different mask formats
without re-running segmentation. It shows the functionality of the converter module
which allows users to convert segmentation masks between different formats (npy, png, 
yolo, overlay) without having to re-run the segmentation process.

Prerequisites:
- To run this example with the default paths, first run basic_usage.py to generate
  the necessary segmentation masks.
- Alternatively, modify the paths below to point to your own mask files.
"""

import os
import sys
from cellsegkit import convert_mask_format

# Create sample mask if needed for demonstration
def create_sample_mask(output_dir):
    """Create a simple sample mask for demonstration purposes."""
    import numpy as np
    from cellsegkit.exporter import save_mask_as_npy, save_mask_as_png

    # Create a simple binary mask (100x100 with a circle in the middle)
    mask = np.zeros((100, 100), dtype=np.uint8)
    y, x = np.ogrid[-50:50, -50:50]
    mask[y**2 + x**2 <= 30**2] = 1

    # Save as NPY and PNG
    os.makedirs(os.path.join(output_dir, "sample"), exist_ok=True)
    npy_path = os.path.join(output_dir, "sample", "sample_mask.npy")
    png_path = os.path.join(output_dir, "sample", "sample_mask.png")

    save_mask_as_npy(mask, npy_path)
    save_mask_as_png(mask, png_path)

    return npy_path, png_path

# Define input and output directories
# You can modify these paths to point to your own files
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "..", "output")
output_dir = os.path.join(base_dir, "..", "output", "mask_conversion")
os.makedirs(output_dir, exist_ok=True)

# Check if we need to create sample masks for demonstration
use_sample_masks = False
if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
    print("Input directory not found. Creating sample masks for demonstration.")
    use_sample_masks = True
    sample_dir = os.path.join(base_dir, "..", "output", "sample")
    os.makedirs(sample_dir, exist_ok=True)
    npy_mask_path, png_mask_path = create_sample_mask(os.path.join(base_dir, ".."))
else:
    # Try to find existing mask files from basic_usage.py output
    npy_mask_path = None
    png_mask_path = None

    # Look for NPY files
    npy_dir = os.path.join(input_dir, "data_only", "npy")
    if os.path.exists(npy_dir):
        npy_files = [f for f in os.listdir(npy_dir) if f.endswith('.npy')]
        if npy_files:
            npy_mask_path = os.path.join(npy_dir, npy_files[0])

    # Look for PNG files
    png_dir = os.path.join(input_dir, "visualization_only", "png")
    if os.path.exists(png_dir):
        png_files = [f for f in os.listdir(png_dir) if f.endswith('.png')]
        if png_files:
            png_mask_path = os.path.join(png_dir, png_files[0])

    # If we couldn't find existing masks, create sample ones
    if not npy_mask_path or not png_mask_path:
        print("Existing mask files not found. Creating sample masks for demonstration.")
        use_sample_masks = True
        npy_mask_path, png_mask_path = create_sample_mask(os.path.join(base_dir, ".."))

# For examples that need an original image
# Create a simple image if we're using sample masks
original_image_path = None
if use_sample_masks:
    import numpy as np
    import cv2

    # Create a simple RGB image
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 64  # Red channel
    img[:, :, 1] = 128  # Green channel
    img[:, :, 2] = 192  # Blue channel

    # Add some features
    cv2.circle(img, (50, 50), 30, (255, 0, 0), 2)
    cv2.rectangle(img, (20, 20), (80, 80), (0, 255, 0), 1)

    # Save the image
    original_image_path = os.path.join(output_dir, "sample_image.png")
    cv2.imwrite(original_image_path, img)
else:
    # Try to find an original image in the dataset directory
    dataset_dir = os.path.join(base_dir, "..", "dataset")
    if os.path.exists(dataset_dir):
        for root, _, files in os.walk(dataset_dir):
            for file in files:
                if file.endswith(('.png', '.jpg', '.tif', '.tiff')):
                    original_image_path = os.path.join(root, file)
                    break
            if original_image_path:
                break

    # If we couldn't find an original image, create a sample one
    if not original_image_path:
        import numpy as np
        import cv2

        # Create a simple RGB image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 64  # Red channel
        img[:, :, 1] = 128  # Green channel
        img[:, :, 2] = 192  # Blue channel

        # Add some features
        cv2.circle(img, (50, 50), 30, (255, 0, 0), 2)
        cv2.rectangle(img, (20, 20), (80, 80), (0, 255, 0), 1)

        # Save the image
        original_image_path = os.path.join(output_dir, "sample_image.png")
        cv2.imwrite(original_image_path, img)

print(f"Using mask files:\n  NPY: {npy_mask_path}\n  PNG: {png_mask_path}")
print(f"Using original image: {original_image_path}")

# Example 1: Convert .npy mask to PNG
print("\n--- Example 1: Convert .npy mask to PNG ---")
convert_mask_format(
    mask_path=npy_mask_path,
    output_format="png",
    output_path=os.path.join(output_dir, "npy_to_png.png"),
)

# Example 2: Convert .npy mask to YOLO format
print("\n--- Example 2: Convert .npy mask to YOLO format ---")
convert_mask_format(
    mask_path=npy_mask_path,
    output_format="yolo",
    output_path=os.path.join(output_dir, "npy_to_yolo.txt"),
<<<<<<< HEAD
    original_image_path="dataset/train/image1.tif",
=======
    original_image_path=original_image_path
>>>>>>> origin/dev
)

# Example 3: Convert .png mask to overlay visualization
print("\n--- Example 3: Convert .png mask to overlay visualization ---")
convert_mask_format(
    mask_path=png_mask_path,
    output_format="overlay",
    output_path=os.path.join(output_dir, "png_to_overlay.png"),
<<<<<<< HEAD
    original_image_path="dataset/train/image1.tif",
=======
    original_image_path=original_image_path
>>>>>>> origin/dev
)

# Example 4: Convert .png mask to .npy format
print("\n--- Example 4: Convert .png mask to .npy format ---")
convert_mask_format(
    mask_path=png_mask_path,
    output_format="npy",
    output_path=os.path.join(output_dir, "png_to_npy.npy"),
)

print(f"\nMask conversion complete. Results saved to {output_dir}")
print("\nThis example demonstrated how to use the converter module to convert between different mask formats:")
print("1. NPY to PNG: Binary mask saved as an image file")
print("2. NPY to YOLO: Mask converted to YOLO annotation format (bounding boxes)")
print("3. PNG to Overlay: Mask overlaid on the original image for visualization")
print("4. PNG to NPY: Image mask converted to numpy array format for further processing")
