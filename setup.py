from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open(
    os.path.join(os.path.abspath(os.path.dirname(__file__)), "README.md"),
    encoding="utf-8",
) as f:
    long_description = f.read()

# This setup.py file is configured to support both local installation (pip install .)
# and installation from GitHub (pip install git+https://github.com/nazarzharskyi/cryobiology3.git).
#
# Core dependencies are installed by default. Optional dependencies can be installed using:
# - pip install ".[cellsam]" - For CellSAM support
# - pip install ".[yolo]" - For YOLO format support
# - pip install ".[gpu]" - For GPU monitoring
# - pip install ".[dev]" - For development tools
# - pip install ".[all]" - For all optional dependencies
#
# When installing from GitHub, use:
# pip install "git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[all]"

setup(
    name="cellsegkit",
    version="0.1.0",
    description="A toolkit for cell segmentation using Cellpose and CellSAM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fedir Yarovyi",
    author_email="fedor.yarovoyi2048@gmail.com",
    url="https://github.com/nazarzharskyi/cryobiology3",
    download_url="https://github.com/nazarzharskyi/cryobiology3/archive/refs/tags/v0.1.0.tar.gz",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "cellpose>=2.2.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "pillow>=8.0.0",
        "matplotlib>=3.3.0",
        "scikit-image>=0.18.0",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "cellsam": [
            "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf",
            "cellSAM",
        ],
        "yolo": ["ultralytics>=8.0.0"],
        "gpu": ["pynvml>=11.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
        "all": [
            "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git@dca509fe793f601edb92606367a655c15ac00fdf",
            "cellSAM",
            "ultralytics>=8.0.0",
            "pynvml>=11.0.0",
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/nazarzharskyi/cryobiology3/issues",
        "Documentation": "https://github.com/nazarzharskyi/cryobiology3",
        "Source Code": "https://github.com/nazarzharskyi/cryobiology3",
    },
)
