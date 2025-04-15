import os


def find_images(input_dir, extensions=('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
    """
    Recursively finds all image files in a directory.

    :param input_dir: Path to the root directory to search
    :param extensions: Tuple of file extensions to include
    :return: List of absolute paths to image files
    """
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def get_relative_output_path(input_path, input_dir, output_dir, suffix='_overlay.png'):
    """
    Generates the output path preserving the folder structure of input_dir.

    :param input_path: Path to the input image
    :param input_dir: Root input directory
    :param output_dir: Root output directory
    :param suffix: Suffix to add to the output file name
    :return: Path to the output file
    """
    relative_path = os.path.relpath(input_path, input_dir)
    base_name, _ = os.path.splitext(relative_path)
    output_path = os.path.join(output_dir, base_name + suffix)
    return output_path
