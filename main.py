#!/usr/bin/env python3
"""
Docker container entry point for CellSegKit.

This script serves as the main entry point for the Docker container.
It reads configuration from environment variables and runs the cell segmentation
pipeline on the mounted input directory.
"""

import os
import sys
import logging
from cellsegkit import SegmenterFactory, run_segmentation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def main():
    """Run the cell segmentation pipeline with configuration from environment variables."""
    # Get configuration from environment variables
    model_type = os.environ.get('MODEL_TYPE', 'cyto')
    export_formats_str = os.environ.get('EXPORT_FORMATS', 'overlay,npy,png,yolo')
    use_gpu = os.environ.get('USE_GPU', 'true').lower() == 'true'
    input_dir = os.environ.get('INPUT_DIR', '/app/images')
    output_dir = os.environ.get('OUTPUT_DIR', '/app/results')
    
    # Parse export formats
    export_formats = tuple(format.strip() for format in export_formats_str.split(','))
    
    logger.info(f"Starting cell segmentation with the following configuration:")
    logger.info(f"  - Model type: {model_type}")
    logger.info(f"  - Export formats: {export_formats}")
    logger.info(f"  - GPU enabled: {use_gpu}")
    logger.info(f"  - Input directory: {input_dir}")
    logger.info(f"  - Output directory: {output_dir}")
    
    # Check if input directory exists and contains files
    if not os.path.exists(input_dir):
        logger.error(f"Input directory '{input_dir}' does not exist.")
        sys.exit(1)
    
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    if not files:
        logger.warning(f"Input directory '{input_dir}' is empty. No files to process.")
        sys.exit(0)
    
    logger.info(f"Found {len(files)} files in the input directory.")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create a segmenter with the specified model type
        logger.info(f"Creating segmenter with model type '{model_type}'...")
        segmenter = SegmenterFactory.create(
            model_type=model_type,
            use_gpu=use_gpu
        )
        
        # Run segmentation
        logger.info("Running segmentation...")
        run_segmentation(
            segmenter=segmenter,
            input_dir=input_dir,
            output_dir=output_dir,
            export_formats=export_formats
        )
        
        logger.info(f"Segmentation complete. Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during segmentation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()