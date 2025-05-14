# CellSegKit GUI Demonstration

This GUI application demonstrates the key features of the CellSegKit library in an interactive way.

## Features

The GUI demonstrates the following features of CellSegKit:

- **Model Selection**: Choose between different segmentation models (cyto, nuclei, cellpose, cellsam)
- **GPU Acceleration**: Toggle GPU usage for faster processing
- **Directory Selection**: Select input and output directories
- **Export Format Selection**: Choose which formats to export (overlay, NumPy, PNG, YOLO)
- **Image Visualization**: View original images and segmentation results side by side
- **Result Navigation**: Browse through multiple images and their segmentation results
- **Mask Conversion**: Convert between different mask formats (overlay, NumPy, PNG, YOLO) without re-running segmentation

## Requirements

In addition to the CellSegKit dependencies, this GUI requires:
- tkinter (usually included with Python)
- matplotlib
- PIL (Pillow)
- numpy

You can install these with:
```bash
pip install matplotlib pillow numpy
```

## Usage

1. Run the GUI demo:
   ```bash
   python examples/gui_demo.py
   ```

2. Select a model type from the dropdown menu (cyto, nuclei, cellpose, cellsam)

3. Toggle GPU usage if you have a compatible GPU

4. Select input directory containing images to segment

5. Select output directory where results will be saved

6. Choose which export formats you want:
   - Overlay: Visual overlay of segmentation boundaries on the original image
   - NumPy: NumPy array for further processing in Python
   - PNG: Indexed PNG mask for use in other software
   - YOLO: YOLO format annotations for object detection tasks

7. Click "Load Images" to load images from the input directory

8. Click "Run Segmentation" to process the images

9. After segmentation completes, use the "Previous" and "Next" buttons to navigate through the results

## Workflows

### Segmentation Workflow

The typical segmentation workflow demonstrated by this GUI is:

1. **Configure**: Set up the segmentation parameters (model, GPU usage, export formats)
2. **Select Directories**: Choose where to find images and where to save results
3. **Load Images**: Load the images to be processed
4. **Run Segmentation**: Process the images using the selected model
5. **View Results**: Examine the segmentation results

This workflow mirrors the programmatic usage of CellSegKit but provides a visual interface for easier interaction.

### Mask Conversion Workflow

The GUI also demonstrates the mask conversion workflow:

1. **Select Mask File**: Choose a mask file (.npy or .png) to convert
2. **Select Original Image**: If converting to overlay or YOLO format, select the original image
3. **Choose Output Format**: Select the desired output format (overlay, NumPy, PNG, YOLO)
4. **Set YOLO Class ID**: If converting to YOLO format, set the class ID (default is 0)
5. **Specify Output Path**: Choose where to save the converted file
6. **Convert Mask**: Process the conversion
7. **View Result**: Examine the conversion result in the preview panel

This demonstrates how to use the converter module to transform masks between different formats without re-running segmentation.

## Notes

- The GUI runs segmentation in a background thread to prevent the interface from freezing
- If you select the CellSAM model, make sure you have the required dependencies installed
- For large datasets, processing may take some time depending on your hardware
