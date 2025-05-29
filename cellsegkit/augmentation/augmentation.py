"""
Image augmentation module for CellSegKit.

This module provides various augmentation techniques for cell images
to improve segmentation quality and model robustness.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union, Callable
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from dataclasses import dataclass
import random
from abc import ABC, abstractmethod


@dataclass
class AugmentationConfig:
    """Configuration for image augmentations."""

    # Geometric transformations
    rotation_range: Tuple[float, float] = (-45, 45)
    flip_horizontal: bool = True
    flip_vertical: bool = True
    zoom_range: Tuple[float, float] = (0.8, 1.2)
    shear_range: float = 0.2

    # Intensity transformations
    brightness_range: Tuple[float, float] = (0.7, 1.3)
    contrast_range: Tuple[float, float] = (0.7, 1.3)
    gamma_range: Tuple[float, float] = (0.7, 1.3)

    # Noise and blur
    gaussian_noise_var: float = 0.01
    gaussian_blur_sigma: Tuple[float, float] = (0, 2.0)

    # Elastic deformation
    elastic_transform: bool = True
    elastic_alpha: float = 120
    elastic_sigma: float = 9

    # Advanced augmentations
    clahe: bool = True
    clahe_clip_limit: float = 2.0

    # Probability of applying each augmentation
    augmentation_probability: float = 0.5

    # Number of augmented versions to generate
    num_augmentations: int = 1


class BaseAugmentor(ABC):
    """Base class for image augmentors."""

    @abstractmethod
    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply augmentation to an image."""
        pass

    def __call__(self, image: np.ndarray, **kwargs) -> np.ndarray:
        return self.apply(image, **kwargs)


class GeometricAugmentor(BaseAugmentor):
    """Handles geometric transformations."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply random geometric transformations."""
        augmented = image.copy()

        # Random rotation
        if random.random() < self.config.augmentation_probability:
            angle = random.uniform(*self.config.rotation_range)
            augmented = self._rotate_image(augmented, angle)

        # Random horizontal flip
        if self.config.flip_horizontal and random.random() < 0.5:
            augmented = np.fliplr(augmented)

        # Random vertical flip
        if self.config.flip_vertical and random.random() < 0.5:
            augmented = np.flipud(augmented)

        # Random zoom
        if random.random() < self.config.augmentation_probability:
            zoom_factor = random.uniform(*self.config.zoom_range)
            augmented = self._zoom_image(augmented, zoom_factor)

        # Random shear
        if (
            self.config.shear_range > 0
            and random.random() < self.config.augmentation_probability
        ):
            shear_factor = random.uniform(
                -self.config.shear_range, self.config.shear_range
            )
            augmented = self._shear_image(augmented, shear_factor)

        return augmented

    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        if len(image.shape) == 3:
            return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        else:
            return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    def _zoom_image(self, image: np.ndarray, zoom_factor: float) -> np.ndarray:
        """Zoom in/out of image."""
        h, w = image.shape[:2]
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Crop or pad to original size
        if zoom_factor > 1:
            # Crop center
            start_h = (new_h - h) // 2
            start_w = (new_w - w) // 2
            return resized[start_h : start_h + h, start_w : start_w + w]
        else:
            # Pad with reflection
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            if len(image.shape) == 3:
                padded = np.zeros_like(image)
                padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
            else:
                padded = np.zeros_like(image)
                padded[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized
            return padded

    def _shear_image(self, image: np.ndarray, shear_factor: float) -> np.ndarray:
        """Apply shear transformation."""
        h, w = image.shape[:2]
        M = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
        return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


class IntensityAugmentor(BaseAugmentor):
    """Handles intensity/color transformations."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply random intensity transformations."""
        augmented = image.copy()

        # Normalize to 0-1 range for processing
        if augmented.dtype == np.uint8:
            augmented = augmented.astype(np.float32) / 255.0
            was_uint8 = True
        else:
            was_uint8 = False

        # Random brightness
        if random.random() < self.config.augmentation_probability:
            brightness_factor = random.uniform(*self.config.brightness_range)
            augmented = augmented * brightness_factor

        # Random contrast
        if random.random() < self.config.augmentation_probability:
            contrast_factor = random.uniform(*self.config.contrast_range)
            mean = np.mean(augmented)
            augmented = (augmented - mean) * contrast_factor + mean

        # Random gamma correction
        if random.random() < self.config.augmentation_probability:
            gamma = random.uniform(*self.config.gamma_range)
            augmented = np.power(augmented, gamma)

        # Clip values and convert back if needed
        augmented = np.clip(augmented, 0, 1)
        if was_uint8:
            augmented = (augmented * 255).astype(np.uint8)

        return augmented


class NoiseAugmentor(BaseAugmentor):
    """Handles noise and blur augmentations."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply noise and blur augmentations."""
        augmented = image.copy()

        # Gaussian noise
        if (
            self.config.gaussian_noise_var > 0
            and random.random() < self.config.augmentation_probability
        ):
            noise = np.random.normal(0, self.config.gaussian_noise_var, augmented.shape)
            if augmented.dtype == np.uint8:
                augmented = np.clip(augmented.astype(np.float32) / 255.0 + noise, 0, 1)
                augmented = (augmented * 255).astype(np.uint8)
            else:
                augmented = np.clip(augmented + noise, 0, 1)

        # Gaussian blur
        if random.random() < self.config.augmentation_probability:
            sigma = random.uniform(*self.config.gaussian_blur_sigma)
            if sigma > 0:
                augmented = cv2.GaussianBlur(augmented, (0, 0), sigma)

        return augmented


class ElasticAugmentor(BaseAugmentor):
    """Handles elastic deformations."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply elastic deformation."""
        if (
            not self.config.elastic_transform
            or random.random() > self.config.augmentation_probability
        ):
            return image

        h, w = image.shape[:2]

        # Generate random displacement fields
        dx = np.random.rand(h, w) * 2 - 1
        dy = np.random.rand(h, w) * 2 - 1

        # Smooth the displacement fields
        dx = (
            cv2.GaussianBlur(dx, (0, 0), self.config.elastic_sigma)
            * self.config.elastic_alpha
        )
        dy = (
            cv2.GaussianBlur(dy, (0, 0), self.config.elastic_sigma)
            * self.config.elastic_alpha
        )

        # Create mesh grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        x_new = (x + dx).astype(np.float32)
        y_new = (y + dy).astype(np.float32)

        # Apply remapping
        if len(image.shape) == 3:
            augmented = cv2.remap(
                image, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )
        else:
            augmented = cv2.remap(
                image, x_new, y_new, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
            )

        return augmented


class CLAHEAugmentor(BaseAugmentor):
    """Contrast Limited Adaptive Histogram Equalization."""

    def __init__(self, config: AugmentationConfig):
        self.config = config

    def apply(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply CLAHE."""
        if (
            not self.config.clahe
            or random.random() > self.config.augmentation_probability
        ):
            return image

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image_uint8 = (np.clip(image, 0, 1) * 255).astype(np.uint8)
        else:
            image_uint8 = image.copy()

        # Apply CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit, tileGridSize=(8, 8)
        )

        if len(image_uint8.shape) == 3:
            # Apply to each channel
            augmented = np.zeros_like(image_uint8)
            for i in range(image_uint8.shape[2]):
                augmented[:, :, i] = clahe.apply(image_uint8[:, :, i])
        else:
            augmented = clahe.apply(image_uint8)

        # Convert back to original dtype
        if image.dtype != np.uint8:
            augmented = augmented.astype(np.float32) / 255.0

        return augmented


class ImageAugmentor:
    """Main augmentor class that combines all augmentation techniques."""

    def __init__(self, config: Optional[AugmentationConfig] = None):
        """Initialize augmentor with configuration."""
        self.config = config or AugmentationConfig()

        # Initialize individual augmentors
        self.geometric = GeometricAugmentor(self.config)
        self.intensity = IntensityAugmentor(self.config)
        self.noise = NoiseAugmentor(self.config)
        self.elastic = ElasticAugmentor(self.config)
        self.clahe = CLAHEAugmentor(self.config)

        # Order of augmentations
        self.augmentors = [
            self.geometric,
            self.elastic,
            self.intensity,
            self.noise,
            self.clahe,
        ]

    def augment(
        self, image: np.ndarray, num_augmentations: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Generate augmented versions of an image.

        Args:
            image: Input image
            num_augmentations: Number of augmented versions to generate
                              (if None, uses config value)

        Returns:
            List of augmented images (including the original)
        """
        num_augs = num_augmentations or self.config.num_augmentations
        augmented_images = [image]  # Include original

        for _ in range(num_augs):
            aug_image = image.copy()

            # Apply augmentations in sequence
            for augmentor in self.augmentors:
                aug_image = augmentor(aug_image)

            augmented_images.append(aug_image)

        return augmented_images

    def augment_batch(
        self, images: List[np.ndarray], num_augmentations: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Augment a batch of images.

        Args:
            images: List of input images
            num_augmentations: Number of augmented versions per image

        Returns:
            List of all images (originals + augmented)
        """
        all_images = []

        for image in images:
            augmented = self.augment(image, num_augmentations)
            all_images.extend(augmented)

        return all_images


# Convenience functions
def create_augmentor(preset: str = "default") -> ImageAugmentor:
    """
    Create an augmentor with preset configurations.

    Args:
        preset: Configuration preset ("default", "light", "heavy", "geometric_only")

    Returns:
        Configured ImageAugmentor instance
    """
    if preset == "light":
        config = AugmentationConfig(
            rotation_range=(-15, 15),
            zoom_range=(0.9, 1.1),
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            gaussian_noise_var=0.005,
            elastic_transform=False,
            augmentation_probability=0.3,
        )
    elif preset == "heavy":
        config = AugmentationConfig(
            rotation_range=(-90, 90),
            zoom_range=(0.5, 1.5),
            brightness_range=(0.5, 1.5),
            contrast_range=(0.5, 1.5),
            gaussian_noise_var=0.02,
            elastic_transform=True,
            augmentation_probability=0.8,
            num_augmentations=3,
        )
    elif preset == "geometric_only":
        config = AugmentationConfig(
            brightness_range=(1.0, 1.0),
            contrast_range=(1.0, 1.0),
            gaussian_noise_var=0,
            clahe=False,
            elastic_transform=False,
        )
    else:  # default
        config = AugmentationConfig()

    return ImageAugmentor(config)


# Integration with existing pipeline
def augment_and_segment(
    segmenter,
    image: np.ndarray,
    augmentor: Optional[ImageAugmentor] = None,
    aggregate_masks: bool = True,
) -> Union[np.ndarray, List[np.ndarray]]:
    """
    Augment an image and run segmentation on all versions.

    Args:
        segmenter: Segmentation model instance
        image: Input image
        augmentor: ImageAugmentor instance (if None, no augmentation)
        aggregate_masks: If True, average the masks; if False, return all masks

    Returns:
        Segmentation mask(s)
    """
    if augmentor is None:
        # No augmentation, just segment
        return segmenter.segment(image)

    # Generate augmented images
    augmented_images = augmentor.augment(image)

    # Segment all versions
    masks = []
    for aug_image in augmented_images:
        mask = segmenter.segment(aug_image)
        masks.append(mask)

    if aggregate_masks:
        # Average the masks for more robust segmentation
        averaged_mask = np.mean(masks, axis=0)
        # Threshold to get binary mask
        return (averaged_mask > 0.5).astype(np.uint8)
    else:
        return masks


# Modified run_segmentation function with augmentation support
def run_segmentation_with_augmentation(
    segmenter,
    input_dir: str,
    output_dir: str,
    export_formats: Tuple[str, ...] = ("overlay", "npy", "png", "yolo"),
    augmentor: Optional[Union[ImageAugmentor, str]] = None,
    save_augmented_images: bool = False,
):
    """
    Run segmentation with optional augmentation.

    Args:
        segmenter: Segmentation model instance
        input_dir: Input directory containing images
        output_dir: Output directory for results
        export_formats: Formats to export
        augmentor: ImageAugmentor instance or preset string
        save_augmented_images: Whether to save augmented images
    """
    # Create augmentor if string preset is provided
    if isinstance(augmentor, str):
        augmentor = create_augmentor(augmentor)

    # Import the necessary functions from the importer module
    from importer import find_images, load_image_with_metadata, get_relative_output_path

    # Find all images
    image_paths = find_images(input_dir)

    for image_path in image_paths:
        # Load image
        image, metadata = load_image_with_metadata(image_path)

        # Get output path
        output_path = get_relative_output_path(image_path, input_dir, output_dir)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        if augmentor:
            # Augment and segment
            mask = augment_and_segment(
                segmenter, image, augmentor, aggregate_masks=True
            )

            # Optionally save augmented images
            if save_augmented_images:
                augmented_images = augmentor.augment(image)
                for i, aug_img in enumerate(augmented_images):
                    aug_path = output_path.replace("_output", f"_aug_{i}")
                    Image.fromarray(aug_img).save(aug_path)
        else:
            # Regular segmentation without augmentation
            mask = segmenter.segment(image)

        # Export results in requested formats
        export_segmentation_results(image, mask, output_path, export_formats)
