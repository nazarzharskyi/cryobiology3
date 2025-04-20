"""
CellSegKit: A toolkit for cell segmentation using Cellpose and CellSAM models.

This package provides tools for loading segmentation models, importing images,
running segmentation, exporting results in various formats, and converting between
mask formats without re-running segmentation.
"""

__version__ = "0.1.0"
__author__ = "Fedir Yarovyi"
__license__ = "MIT"

from cellsegkit.loader import SegmenterFactory
from cellsegkit.pipeline import run_segmentation
from cellsegkit.converter import convert_mask_format

# Import utils module for internal use
import cellsegkit.utils

__all__ = ["SegmenterFactory", "run_segmentation", "convert_mask_format"]
