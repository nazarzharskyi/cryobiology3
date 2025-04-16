import os
import cv2
from cellpose import models, core
import numpy as np
from cellSAM import segment_cellular_image

class SegmenterFactory:
    @staticmethod
    def create(model_type: str, use_gpu=True, sam_checkpoint_path=None):
        """
        Factory method to instantiate the correct segmenter based on model_type.
        """
        if model_type.lower() in ["cyto", "nuclei", "cellpose"]:
            return CellposeSegmenter(model_type=model_type, use_gpu=use_gpu)
        elif model_type.lower() == "cellsam":
            return CellSAMSegmenter(use_gpu=use_gpu)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

class CellposeSegmenter:
    """
    A class used for cell segmentation using the Cellpose deep learning model. It supports
    GPU acceleration and provides functionalities to load and preprocess images, perform
    segmentation, and visualize segmentation results.

    :ivar model_type: Specifies the model type to be used by Cellpose, e.g., 'cyto' for
        cytoplasm segmentation or 'nuclei' for nuclei segmentation.
    :type model_type: str
    :ivar use_gpu: Indicates whether the GPU should be used for model inference. This
        depends on the availability of GPU hardware.
    :type use_gpu: bool
    :ivar model: An instance of the CellposeModel initialized with the specified model
        type and enabling GPU usage if applicable.
    :type model: CellposeModel
    """
    def __init__(self, model_type='cyto', use_gpu=True):
        """
        Initialize the CellposeSegmenter with specified model type ('cyto' or 'nuclei').
        Optionally enable GPU acceleration.
        """
        self.model_type = model_type
        self.use_gpu = use_gpu and core.use_gpu()
        self.model = models.CellposeModel(gpu=self.use_gpu, model_type=self.model_type)

    def load_image(self, file_path):
        """
        Load an image from a file path, supporting grayscale or RGB formats.
        :param file_path: Path to the input image.
        :return: Loaded image as a numpy array.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Image file '{file_path}' not found.")
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Failed to load the image. Please ensure it's a valid file.")
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] > 3:  # Remove alpha channel
            image = image[:, :, :3]
        return image

    def segment(self, image):
        """
        Run the Cellpose model on the provided image.
        :param image: Input image as a numpy array.
        :return: Segmentation mask as a numpy array.
        """
        masks, _, _ = self.model.eval(image, diameter=None, channels=[0, 0])
        return masks

class CellSAMSegmenter:
    """
    CellSAM segmenter using the official `segment_cellular_image()` function.
    """

    def __init__(self, use_gpu=True):
        self.device = 'cuda' if use_gpu and self._cuda_available() else 'cpu'

    def load_image(self, file_path):
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Failed to load image: {file_path}")
        if len(image.shape) == 2:
            return image
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def segment(self, image):
        mask, _, _ = segment_cellular_image(image, device=self.device)
        return mask

    def _cuda_available(self):
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False