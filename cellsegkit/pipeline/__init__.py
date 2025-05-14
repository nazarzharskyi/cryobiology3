"""
Pipeline module for cell segmentation.

This module provides a unified workflow for cell segmentation, combining
model loading, image importing, segmentation, and result exporting.
"""

from cellsegkit.pipeline.pipeline import run_segmentation

__all__ = ["run_segmentation"]
