"""
System utility functions for CellSegKit.

This module provides functions for monitoring system resources like CPU and GPU utilization.
"""

import psutil
from typing import Union

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml

    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except (ImportError, Exception):
    NVML_AVAILABLE = False


def get_cpu_utilization() -> float:
    """
    Get current CPU utilization as a percentage.

    Returns:
        CPU utilization percentage (0-100)
    """
    return psutil.cpu_percent(interval=0.1)


def get_gpu_utilization() -> Union[float, None]:
    """
    Get current GPU utilization as a percentage if available.

    Returns:
        GPU utilization percentage (0-100) or None if not available
    """
    if TORCH_AVAILABLE and torch.cuda.is_available():
        # Try using NVML first if available
        if NVML_AVAILABLE:
            try:
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    # Get the first GPU (index 0)
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    return util.gpu  # This is already a percentage (0-100)
            except Exception:
                pass

        # Fallback to simpler check if CUDA is being used
        return 100.0 if torch.cuda.memory_allocated() > 0 else 0.0

    return None
