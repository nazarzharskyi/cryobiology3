from loader.model_loader import SegmenterFactory
from pipeline.pipeline import run_segmentation

segmenter = SegmenterFactory.create(
    model_type="cellsam",  # или "cyto", "nuclei"
    use_gpu=True,
)

run_segmentation(
    segmenter=segmenter,
    input_dir=r"C:\Users\KusFedots\PycharmProjects\cryobiology3\dataset",
    output_dir=r"C:\Users\KusFedots\PycharmProjects\cryobiology3\output",
    export_formats=("overlay", "npy", "png", "yolo")
)
