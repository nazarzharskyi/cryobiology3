"""
Model loader module for cell segmentation.

This module provides classes for loading and using different segmentation models,
including Cellpose and CellSAM.
"""

import os
import cv2
from cellpose import models, core
import numpy as np
from abc import ABC, abstractmethod

# Try to import cellSAM, but don't fail if it's not installed
try:
    from cellSAM import segment_cellular_image
    CELLSAM_AVAILABLE = True
except ImportError:
    CELLSAM_AVAILABLE = False


def load_image_file(file_path, grayscale=False):
    """
    Load an image from a file path, supporting grayscale or RGB formats.

    Args:
        file_path: Path to the input image
        grayscale: Whether to convert the image to grayscale

    Returns:
        Loaded image as a numpy array

    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image can't be loaded
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Image file '{file_path}' not found.")

    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to load image: {file_path}")

    # Handle different image formats
    if len(image.shape) == 2:  # Already grayscale
        if not grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3:
        if image.shape[2] > 3:  # Remove alpha channel
            image = image[:, :, :3]
        if grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


class BaseSegmenter(ABC):
    """Base class for all segmenters."""

    @abstractmethod
    def segment(self, image):
        """
        Run segmentation on the provided image.

        Args:
            image: Input image as a numpy array

        Returns:
            Segmentation mask as a numpy array
        """
        pass


class SegmenterFactory:
    """Factory class for creating segmentation model instances."""

    @staticmethod
    def create(model_type: str, use_gpu=True, sam_checkpoint_path=None):
        """
        Factory method to instantiate the correct segmenter based on model_type.

        Args:
            model_type: Type of model to create ("cyto", "nuclei", "cellpose", or "cellsam")
            use_gpu: Whether to use GPU acceleration if available
            sam_checkpoint_path: Path to SAM checkpoint file (for CellSAM only)

        Returns:
            An instance of a segmenter class

        Raises:
            ValueError: If an unknown model_type is provided
            ImportError: If the required dependencies for the selected model are not installed
        """
        if model_type.lower() in ["cyto", "nuclei", "cellpose"]:
            return CellposeSegmenter(model_type=model_type, use_gpu=use_gpu)
        elif model_type.lower() == "cellsam":
            if not CELLSAM_AVAILABLE:
                raise ImportError(
                    "cellSAM is not installed. Install it with: "
                    "pip install 'cellsegkit[cellsam]' or "
                    "pip install 'git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[cellsam]'"
                )
            return CellSAMSegmenter(use_gpu=use_gpu)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")


class CellposeSegmenter(BaseSegmenter):
    """
    A class used for cell segmentation using the Cellpose deep learning model.

    It supports GPU acceleration and provides functionalities to load and preprocess images,
    perform segmentation, and visualize segmentation results.
    """

    def __init__(self, model_type='cyto', use_gpu=True):
        """
        Initialize the CellposeSegmenter with specified model type ('cyto' or 'nuclei').

        Args:
            model_type: Type of Cellpose model to use ('cyto' or 'nuclei')
            use_gpu: Whether to use GPU acceleration if available
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and core.use_gpu()
        self.model = models.CellposeModel(gpu=self.use_gpu, model_type=self.model_type)

    def load_image(self, file_path):
        """
        Load an image from a file path, supporting grayscale or RGB formats.

        Args:
            file_path: Path to the input image

        Returns:
            Loaded image as a numpy array

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image can't be loaded
        """
        return load_image_file(file_path, grayscale=False)

    def segment(self, image):
        """
        Run the Cellpose model on the provided image.

        Args:
            image: Input image as a numpy array

        Returns:
            Segmentation mask as a numpy array
        """
        masks, _, _ = self.model.eval(image, diameter=None, channels=[0, 0])
        return masks


class CellSAMSegmenter(BaseSegmenter):
    """
    CellSAM segmenter using the official `segment_cellular_image()` function.
    """

    def __init__(self, use_gpu=True):
        """
        Initialize the CellSAMSegmenter.

        Args:
            use_gpu: Whether to use GPU acceleration if available

        Raises:
            ImportError: If cellSAM is not installed
        """
        if not CELLSAM_AVAILABLE:
            raise ImportError(
                "cellSAM is not installed. Install it with: "
                "pip install 'cellsegkit[cellsam]' or "
                "pip install 'git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[cellsam]'"
            )
        self.device = 'cuda' if use_gpu and self._cuda_available() else 'cpu'

    def load_image(self, file_path):
        """
        Load an image from a file path, converting to grayscale for CellSAM.

        Args:
            file_path: Path to the input image

        Returns:
            Loaded image as a numpy array

        Raises:
            FileNotFoundError: If the image file doesn't exist
            ValueError: If the image can't be loaded
        """
        return load_image_file(file_path, grayscale=True)

    def segment(self, image):
        """
        Run the CellSAM model on the provided image.

        Args:
            image: Input image as a numpy array

        Returns:
            Segmentation mask as a numpy array
        """
        if not CELLSAM_AVAILABLE:
            raise ImportError(
                "cellSAM is not installed. Install it with: "
                "pip install 'cellsegkit[cellsam]' or "
                "pip install 'git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[cellsam]'"
            )
        mask, _, _ = segment_cellular_image(image, device=self.device)
        return mask

    def _cuda_available(self):
        """
        Check if CUDA is available for GPU acceleration.

        Returns:
            Boolean indicating if CUDA is available
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
