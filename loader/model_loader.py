import os
import cv2
import matplotlib.pyplot as plt
from cellpose import models, core
from cellpose.io import save_masks
from cellpose.utils import masks_to_outlines
from skimage.segmentation import find_boundaries


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

    def visualize(self, image, masks, output_path):
        """
        Visualize the segmentation result and save it to the output path.
        :param image: Original image as a numpy array.
        :param masks: Segmentation masks as a numpy array.
        :param output_path: Path to save the overlay visualization.
        """
        # Получаем границы масок
        boundaries = find_boundaries(masks, mode='outer')

        # Создаём копию изображения
        overlaid = image.copy()

        # Покрасим границы в красный
        overlaid[boundaries] = [255, 0, 0]

        # Сохраняем изображение
        plt.figure(figsize=(10, 10))
        plt.imshow(overlaid)
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()
