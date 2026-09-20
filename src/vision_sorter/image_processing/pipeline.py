"""Inspectable preprocessing stages shared by the CLI and detector."""
from pathlib import Path
from vision_sorter.image_processing.reader import Image, write_image
from vision_sorter.image_processing.preprocessing import grayscale, gaussian_blur, opening
from vision_sorter.image_processing.thresholding import otsu_threshold
from vision_sorter.image_processing.edges import canny_edges


def preprocess(image: Image, invert: bool = False, destination: Path | None = None) -> dict[str, Image]:
    gray = grayscale(image)
    blurred = gaussian_blur(gray)
    binary = otsu_threshold(blurred, invert)
    stages = {"original": image.copy(), "grayscale": gray, "blurred": blurred,
              "binary": binary, "edges": canny_edges(blurred), "morphology": opening(binary)}
    if destination is not None:
        for name, output in stages.items():
            write_image(destination / f"{name}.png", output)
    return stages
