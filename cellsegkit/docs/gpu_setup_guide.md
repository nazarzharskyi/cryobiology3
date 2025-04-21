# GPU Setup Guide for Cell Segmentation

This guide will help you set up your environment to use GPU acceleration with our cell segmentation tools.

## Verifying GPU Hardware

Before installing any software, verify that your system has a compatible NVIDIA GPU:

1. **Windows**: Open Command Prompt or PowerShell and run:
   ```
   nvidia-smi
   ```

2. **Linux/macOS**: Open Terminal and run:
   ```
   nvidia-smi
   ```

If you see information about your GPU, including the driver version and CUDA version, your GPU is recognized by the system. If you get an error, you may need to install or update your NVIDIA drivers.

## Installing or Updating NVIDIA Drivers

### Windows

1. Visit the [NVIDIA Driver Downloads](https://www.nvidia.com/Download/index.aspx) page
2. Select your GPU model and operating system
3. Download and run the installer
4. Follow the installation wizard instructions

### Linux

For Ubuntu/Debian:
```bash
sudo apt update
sudo apt install nvidia-driver-XXX  # Replace XXX with the latest version number
sudo reboot
```

For other distributions, refer to your package manager or the [NVIDIA documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

## Installing CUDA Toolkit

We recommend CUDA 11.7 or newer for optimal compatibility with PyTorch.

### Windows

1. Visit the [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) page
2. Select your operating system and download the installer
3. Run the installer and follow the instructions

### Linux

For Ubuntu/Debian:
```bash
wget https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run
sudo sh cuda_11.7.0_515.43.04_linux.run
```

Add CUDA to your PATH by adding these lines to your `~/.bashrc` file:
```bash
export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

## Installing PyTorch with CUDA Support

Install PyTorch with CUDA support using pip:

```bash
# For CUDA 11.7
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Verifying Your Setup

To verify that PyTorch can access your GPU, run:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
```

## Fallback Options

If you cannot install a GPU locally, consider these alternatives:

### Using Google Colab

Google Colab provides free GPU access:

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Go to Runtime > Change runtime type > Hardware accelerator > GPU
4. Install our package:
   ```
   !pip install git+https://github.com/nazarzharskyi/cryobiology3.git
   ```

### Using Docker

We provide Docker images with all dependencies pre-installed:

```bash
# Pull the Docker image
docker pull nazarzharskyi/cellsegkit:latest

# Run the container with GPU support
docker run --gpus all -it nazarzharskyi/cellsegkit:latest
```

## Troubleshooting

### Common Issues

1. **"CUDA out of memory"**: Your GPU doesn't have enough VRAM. Try reducing batch sizes or image dimensions.

2. **"CUDA error: no kernel image is available for execution"**: CUDA version mismatch. Make sure your PyTorch CUDA version matches your installed CUDA toolkit.

3. **"CUDA initialization: CUDA driver version is insufficient"**: Update your NVIDIA drivers.

### Getting Help

If you encounter issues not covered here, please:

1. Check our [GitHub Issues](https://github.com/nazarzharskyi/cryobiology3/issues)
2. Open a new issue with details about your environment and the error message

## Recommended Versions

For optimal compatibility, we recommend:

- CUDA Toolkit: 11.7 or 11.8
- PyTorch: 2.0.0 or newer
- NVIDIA Driver: 515.43.04 or newer