"""
Module: model_loader.py
This module provides functionality to load pretrained segmentation models 
for automatic image annotation tasks.
"""

from ultralytics import YOLO


def load_model(model_name="yolov8n-seg.pt", device=None):
    """
    Load a pretrained segmentation model using the Ultralytics YOLOv8 framework.

    Parameters:
        model_name (str): Name or path to the pretrained segmentation model.
                          Default is "yolov8n-seg.pt".
        device (str, optional): Device to run the model on (e.g., "cpu", "cuda").
                                If None, the default device is used.

    Returns:
        YOLO: A YOLO segmentation model object.

    Raises:
        RuntimeError: If the model cannot be loaded.
    
    Example:
    """
    try:
        # Initialize the YOLO model with the specified name or path
        model = YOLO(model_name)

        # If device is specified, switch the model to the given device
        if device:
            model.to(device)

        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model '{model_name}' on device '{device}': {e}")