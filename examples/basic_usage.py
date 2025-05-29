"""
Examples of using the enhanced run_segmentation function with augmentation.
"""

from cellsegkit import SegmenterFactory, run_segmentation
from cellsegkit.augmentation.augmentation import AugmentationConfig

# Example 1: Basic usage - backward compatible
print("Example 1: Standard segmentation (no changes needed)")
segmenter = SegmenterFactory.create(model_type="cyto", use_gpu=True)

run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output"
)

# Example 2: Simple augmentation with preset
print("\nExample 2: With light augmentation preset")
result = run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    augmentation="light"  # Just add this parameter!
)

print(f"Processed: {result['processed']} images")
if result['confidence_stats']:
    print(f"Mean confidence: {result['confidence_stats']['mean']:.3f}")

# Example 3: Custom augmentation configuration
print("\nExample 3: Custom augmentation settings")
result = run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    augmentation=True,  # Enable augmentation
    augmentation_config={
        "rotation_range": (-20, 20),
        "flip_horizontal": True,
        "flip_vertical": False,  # Disable for microscopy
        "brightness_range": (0.9, 1.1),
        "elastic_transform": True,
        "num_augmentations": 3  # Use 3 augmented versions
    },
    save_augmented=True,  # Save augmented images for inspection
    aggregate_method="mean"  # How to combine predictions
)

# Example 4: Heavy augmentation for difficult images
print("\nExample 4: Heavy augmentation for challenging data")
result = run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    augmentation="heavy",
    aggregate_method="majority",  # Use majority voting
    confidence_threshold=0.6,  # Lower threshold for difficult images
    export_formats=("overlay", "npy", "confidence")  # Include confidence maps
)

# Check for low-confidence results
if result['confidence_stats']['below_threshold'] > 0:
    print(f"Warning: {result['confidence_stats']['below_threshold']} images had low confidence")

# Example 5: Using test-time augmentation helper
print("\nExample 5: Test-time augmentation helper")
from cellsegkit.pipeline import run_segmentation_with_tta

result = run_segmentation_with_tta(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    augmentation_preset="default",
    num_augmentations=5,  # Use 5 augmented versions for high accuracy
    export_formats=("overlay", "npy", "confidence", "png")
)

# Example 6: Custom augmentor instance
print("\nExample 6: Custom augmentor with specific requirements")
from cellsegkit.augmentation.augmentation import ImageAugmentor, AugmentationConfig

# Configure for specific imaging conditions
custom_config = AugmentationConfig(
    # Limited rotation for aligned samples
    rotation_range=(-10, 10),
    
    # No flipping for directional features
    flip_horizontal=False,
    flip_vertical=False,
    
    # Compensate for illumination variations
    brightness_range=(0.7, 1.3),
    contrast_range=(0.8, 1.2),
    
    # Add slight noise for robustness
    gaussian_noise_var=0.005,
    
    # Strong elastic deformation for cell variability
    elastic_transform=True,
    elastic_alpha=150,
    elastic_sigma=10,
    
    # Enhance local contrast
    clahe=True,
    clahe_clip_limit=3.0,
    
    # Higher probability for each augmentation
    augmentation_probability=0.7,
    
    # Generate 4 versions
    num_augmentations=4
)

custom_augmentor = ImageAugmentor(custom_config)

result = run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    augmentation=custom_augmentor,
    aggregate_method="mean",
    save_augmented=True
)

# Example 7: Batch processing with memory optimization
print("\nExample 7: Large dataset with memory optimization")
result = run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    augmentation="light",
    tta_batch_size=2,  # Process augmented images in batches of 2
    export_formats=("overlay", "npy")  # Skip confidence maps to save space
)

# Example 8: Unanimous agreement for high precision
print("\nExample 8: High precision mode")
result = run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    augmentation="default",
    aggregate_method="unanimous",  # Only keep regions where all augmentations agree
    export_formats=("overlay", "npy", "confidence")
)

# Example 9: Analyze errors and confidence
print("\nExample 9: Detailed analysis")
result = run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    augmentation="default",
    export_formats=("overlay", "npy", "confidence", "png", "yolo")
)

# Analyze results
print(f"\nAnalysis Results:")
print(f"- Total images: {result['processed'] + len(result['errors'])}")
print(f"- Successfully processed: {result['processed']}")
print(f"- Failed: {len(result['errors'])}")

if result['augmentation_used']:
    print(f"\nAugmentation Statistics:")
    print(f"- Mean confidence: {result['confidence_stats']['mean']:.3f}")
    print(f"- Minimum confidence: {result['confidence_stats']['min']:.3f}")
    print(f"- Low confidence images: {result['confidence_stats']['below_threshold']}")

if result['errors']:
    print(f"\nError Details:")
    for filename, error in result['errors'][:5]:  # Show first 5 errors
        print(f"- {filename}: {error}")

# Example 10: Integration with existing workflow
print("\nExample 10: Integration with existing workflow")

# Your existing code
segmenter = SegmenterFactory.create(model_type="cyto", use_gpu=True)

# Simply add augmentation parameter to existing calls
run_segmentation(
    segmenter=segmenter,
    input_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output",
    export_formats=("overlay", "npy", "png", "yolo"),
    augmentation="light"  # That's it! Augmentation is now enabled
)

# Example 11: Conditional augmentation based on image quality
print("\nExample 11: Conditional augmentation")

def process_with_adaptive_augmentation(segmenter, image_dir, output_dir):
    """Apply different augmentation based on image characteristics."""
    import os
    from cellsegkit.importer import find_images, load_image_with_metadata
    
    # Analyze images first
    images = find_images(image_dir)
    
    # Simple quality check (you can make this more sophisticated)
    low_quality_images = []
    high_quality_images = []
    
    for img_path in images:
        image, _ = load_image_with_metadata(img_path)
        # Simple quality metric (contrast)
        contrast = image.std()
        
        if contrast < 30:  # Low contrast threshold
            low_quality_images.append(img_path)
        else:
            high_quality_images.append(img_path)
    
    # Process with different augmentation levels
    if low_quality_images:
        print(f"Processing {len(low_quality_images)} low quality images with heavy augmentation")
        result_low = run_segmentation(
            segmenter=segmenter,
            input_dir=image_dir,
            output_dir=os.path.join(output_dir, "low_quality"),
            augmentation="heavy",
            aggregate_method="mean"
        )
    
    if high_quality_images:
        print(f"Processing {len(high_quality_images)} high quality images with light augmentation")
        result_high = run_segmentation(
            segmenter=segmenter,
            input_dir=image_dir,
            output_dir=os.path.join(output_dir, "high_quality"),
            augmentation="light",
            aggregate_method="majority"
        )

# Use the adaptive function
process_with_adaptive_augmentation(
    segmenter=segmenter,
    image_dir="C:\\fun\dataset",
    output_dir="C:\\fun\output"
)