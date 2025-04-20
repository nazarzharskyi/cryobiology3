from setuptools import setup, find_packages
import os

# Read the contents of the README file
with open(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="cellsegkit",
    version="0.1.0",
    description="A toolkit for cell segmentation using Cellpose and CellSAM models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Fedir Yarovyi",
    author_email="fedor.yarovoyi2048@gmail.com",
    url="https://github.com/Falian2048/cellsegkit",
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
        "cellsam": ["segment-anything", "cellSAM"],
        "yolo": ["ultralytics"],
        "gpu": ["pynvml>=11.0.0"],
        "dev": [
            "pytest>=6.0.0",
            "black>=21.5b2",
            "isort>=5.9.1",
            "flake8>=3.9.2",
        ],
        "all": [
            "segment-anything", 
            "cellSAM", 
            "ultralytics", 
            "pynvml>=11.0.0"
        ],
    },
    project_urls={
        "Bug Tracker": "https://github.com/Falian2048/cellsegkit/issues",
        "Documentation": "https://github.com/Falian2048/cellsegkit",
        "Source Code": "https://github.com/Falian2048/cellsegkit",
    },
)
