"""
GPU utilities for cell segmentation.

This module provides utilities for detecting and using GPU acceleration
with PyTorch and CUDA for cell segmentation tasks.
"""

import os
import subprocess
import logging
import platform
from typing import Tuple, Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_nvidia_smi() -> bool:
    """
    Check if nvidia-smi command is available and returns successfully.
    
    Returns:
        bool: True if nvidia-smi is available and returns successfully, False otherwise.
    """
    try:
        # Use subprocess with a timeout to avoid hanging
        subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True, 
            timeout=5
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def get_cuda_version_from_nvidia_smi() -> Optional[str]:
    """
    Get CUDA version from nvidia-smi output.
    
    Returns:
        Optional[str]: CUDA version string (e.g., "11.7") or None if not found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            check=True, 
            timeout=5,
            text=True
        )
        
        # Parse the output to find CUDA version
        for line in result.stdout.splitlines():
            if "CUDA Version:" in line:
                return line.split("CUDA Version:")[1].strip()
        return None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

def get_torch_cuda_capabilities() -> Tuple[bool, Optional[str], Optional[List[str]]]:
    """
    Check PyTorch CUDA capabilities and version.
    
    Returns:
        Tuple[bool, Optional[str], Optional[List[str]]]: 
            - bool: Whether CUDA is available through PyTorch
            - Optional[str]: PyTorch CUDA version if available
            - Optional[List[str]]: List of CUDA device names if available
    """
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else None
        
        device_names = []
        if cuda_available:
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_names.append(torch.cuda.get_device_name(i))
        
        return cuda_available, cuda_version, device_names
    except ImportError:
        logger.warning("PyTorch is not installed. GPU acceleration will not be available.")
        return False, None, None

def get_device(prefer_gpu: bool = True) -> 'torch.device':
    """
    Get the appropriate device (CUDA or CPU) based on availability and preference.
    
    Args:
        prefer_gpu (bool): Whether to prefer GPU over CPU if available.
        
    Returns:
        torch.device: The selected device (either 'cuda' or 'cpu').
    """
    try:
        import torch
        
        if prefer_gpu and torch.cuda.is_available():
            return torch.device('cuda')
        return torch.device('cpu')
    except ImportError:
        # If torch is not available, return a string as fallback
        # (though this would likely cause errors elsewhere)
        logger.error("PyTorch is not installed. Returning 'cpu' as device.")
        class FallbackDevice:
            def __str__(self):
                return 'cpu'
        return FallbackDevice()

def check_gpu_availability(verbose: bool = True) -> bool:
    """
    Check if GPU is available and print/log friendly messages.
    
    Args:
        verbose (bool): Whether to print/log messages about GPU availability.
        
    Returns:
        bool: True if GPU is available, False otherwise.
    """
    # Check nvidia-smi
    nvidia_smi_available = check_nvidia_smi()
    cuda_version_from_smi = get_cuda_version_from_nvidia_smi() if nvidia_smi_available else None
    
    # Check PyTorch CUDA capabilities
    torch_cuda_available, torch_cuda_version, device_names = get_torch_cuda_capabilities()
    
    # Determine overall GPU availability
    gpu_available = torch_cuda_available
    
    if verbose:
        if gpu_available:
            cuda_version = torch_cuda_version or cuda_version_from_smi or "Unknown"
            devices_str = ", ".join(device_names) if device_names else "Unknown GPU"
            logger.info(f"✅ GPU found, using CUDA {cuda_version} with {devices_str}")
        else:
            logger.warning(
                "⚠️ No compatible GPU detected—falling back to CPU. "
                "To enable GPU, please install NVIDIA drivers, CUDA toolkit, "
                "and torch with CUDA support."
            )
            
            # Provide more detailed diagnostics
            if not nvidia_smi_available:
                logger.info("   - NVIDIA System Management Interface (nvidia-smi) not found or not working")
            elif not cuda_version_from_smi:
                logger.info("   - CUDA version not detected from nvidia-smi")
            
            if not torch_cuda_available:
                logger.info("   - PyTorch CUDA support is not available")
            
    return gpu_available

def get_gpu_info() -> dict:
    """
    Get detailed information about GPU and CUDA setup.
    
    Returns:
        dict: Dictionary containing GPU and CUDA information.
    """
    info = {
        "platform": platform.platform(),
        "nvidia_smi_available": check_nvidia_smi(),
        "cuda_version_from_smi": get_cuda_version_from_nvidia_smi(),
    }
    
    # Add PyTorch information
    torch_cuda_available, torch_cuda_version, device_names = get_torch_cuda_capabilities()
    info.update({
        "torch_cuda_available": torch_cuda_available,
        "torch_cuda_version": torch_cuda_version,
        "gpu_devices": device_names,
    })
    
    # Try to get more detailed GPU information if available
    if torch_cuda_available:
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["gpu_count"] = torch.cuda.device_count()
            
            # Get memory information for each device
            memory_info = []
            for i in range(info["gpu_count"]):
                torch.cuda.set_device(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory
                memory_info.append({
                    "device": i,
                    "name": torch.cuda.get_device_name(i),
                    "total_memory_bytes": total_memory,
                    "total_memory_gb": round(total_memory / (1024**3), 2)
                })
            info["memory_info"] = memory_info
        except Exception as e:
            info["error_getting_detailed_info"] = str(e)
    
    return info

# Run a check when the module is imported
gpu_available = check_gpu_availability(verbose=True)