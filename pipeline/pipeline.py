from loader.model_loader import CellposeSegmenter
from importer.importer import find_images, get_relative_output_path
import os


def run_segmentation(input_dir, output_dir, model_type="cyto"):
    """
    Run full segmentation pipeline on a folder of images.

    :param input_dir: Root directory of input images
    :param output_dir: Root directory to save results
    :param model_type: Model to use ('cyto' or 'nuclei')
    """
    segmenter = CellposeSegmenter(model_type=model_type, use_gpu=True)
    image_paths = find_images(input_dir)

    for image_path in image_paths:
        try:
            image = segmenter.load_image(image_path)
            masks = segmenter.segment(image)

            out_path = get_relative_output_path(image_path, input_dir, output_dir)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            segmenter.visualize(image, masks, out_path)

            print(f"✅ {image_path} → {out_path}")
        except Exception as e:
            print(f"❌ Failed {image_path}: {e}")
