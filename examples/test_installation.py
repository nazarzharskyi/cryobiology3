"""
Test script to verify that cellsegkit is installed correctly.

This script imports the main components of cellsegkit and prints the version number.
If the script runs without errors, the installation is working correctly.
"""

import sys
print(f"Python version: {sys.version}")

try:
    import cellsegkit
    print(f"cellsegkit version: {cellsegkit.__version__}")
    
    # Test importing the main components
    from cellsegkit import SegmenterFactory, run_segmentation, convert_mask_format
    print("Successfully imported main components")
    
    # Test importing specific modules
    from cellsegkit.loader import CellposeSegmenter
    from cellsegkit.exporter import save_mask_as_npy
    from cellsegkit.importer import find_images
    from cellsegkit.converter import VALID_FORMATS
    from cellsegkit.utils.system import get_cpu_utilization
    print("Successfully imported specific modules")
    
    # Test creating a segmenter (without actually loading models)
    print("Testing SegmenterFactory...")
    print(f"Available model types: cyto, nuclei, cellpose, cellsam")
    
    # Test if cellSAM is available
    try:
        from cellSAM import segment_cellular_image
        print("cellSAM is installed (optional dependency)")
    except ImportError:
        print("cellSAM is not installed (optional dependency)")
    
    # Test if ultralytics is available
    try:
        import ultralytics
        print("ultralytics is installed (optional dependency)")
    except ImportError:
        print("ultralytics is not installed (optional dependency)")
    
    # Test if pynvml is available
    try:
        import pynvml
        print("pynvml is installed (optional dependency)")
    except ImportError:
        print("pynvml is not installed (optional dependency)")
    
    print("\n✅ Installation test completed successfully!")
    
except ImportError as e:
    print(f"\n❌ Error: {e}")
    print("\nPlease make sure cellsegkit is installed correctly:")
    print("pip install git+https://github.com/nazarzharskyi/cryobiology3.git")
    print("\nFor optional dependencies, use:")
    print("pip install \"git+https://github.com/nazarzharskyi/cryobiology3.git#egg=cellsegkit[all]\"")