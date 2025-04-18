"""
CellSegKit: A toolkit for cell segmentation using Cellpose and CellSAM models.

This package provides tools for loading segmentation models, importing images,
running segmentation, and exporting results in various formats.
"""

__version__ = "0.1.0"

from cellsegkit.loader import SegmenterFactory
from cellsegkit.pipeline import run_segmentation

__all__ = ["SegmenterFactory", "run_segmentation"]