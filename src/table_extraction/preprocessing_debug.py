from __future__ import annotations

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .grid_detection import (
    GAUSSIAN_BLUR_KERNEL,
    HORIZONTAL_KERNEL_DIVISOR,
    VERTICAL_KERNEL_DIVISOR,
    extract_blue_rule_mask,
    remove_light_watermark,
)


def build_preprocessing_stages(
    image: Image.Image,
    *,
    watermark_gray_threshold: int = 200,
    horizontal_kernel_divisor: int = HORIZONTAL_KERNEL_DIVISOR,
    vertical_kernel_divisor: int = VERTICAL_KERNEL_DIVISOR,
    dilation_iterations: int = 2,
) -> dict[str, Image.Image | np.ndarray]:
    """Build intermediate images used before line detection."""
    original_rgb = image.convert("RGB")
    original_arr = np.array(original_rgb)
    grayscale = cv2.cvtColor(original_arr, cv2.COLOR_RGB2GRAY)
    watermark_mask = grayscale > watermark_gray_threshold
    cleaned_rgb = remove_light_watermark(original_rgb, gray_threshold=watermark_gray_threshold)
    cleaned_gray = cv2.cvtColor(np.array(cleaned_rgb), cv2.COLOR_RGB2GRAY)
    blurred_gray = cv2.GaussianBlur(cleaned_gray, GAUSSIAN_BLUR_KERNEL, 0)
    _, binary = cv2.threshold(blurred_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    blue_rule_mask = extract_blue_rule_mask(cleaned_rgb)
    combined_binary = cv2.bitwise_or(binary, blue_rule_mask)

    img_height, img_width = combined_binary.shape
    horizontal_kernel_len = max(1, img_width // horizontal_kernel_divisor)
    vertical_kernel_len = max(1, img_height // vertical_kernel_divisor)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_len, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_len))

    horizontal_eroded = cv2.erode(combined_binary.copy(), horizontal_kernel, iterations=1)
    horizontal_morph = cv2.dilate(horizontal_eroded, horizontal_kernel, iterations=dilation_iterations)
    vertical_eroded = cv2.erode(combined_binary.copy(), vertical_kernel, iterations=1)
    vertical_morph = cv2.dilate(vertical_eroded, vertical_kernel, iterations=dilation_iterations)

    return {
        "Original RGB": original_rgb,
        "Grayscale": grayscale,
        "Watermark mask": watermark_mask.astype(np.uint8) * 255,
        "Watermark suppressed": cleaned_rgb,
        "Blurred grayscale": blurred_gray,
        "Otsu binary": binary,
        "Blue rule mask": blue_rule_mask,
        "Combined binary for line detection": combined_binary,
        "Horizontal morphology mask": horizontal_morph,
        "Vertical morphology mask": vertical_morph,
    }


def show_preprocessing_stages(
    image: Image.Image,
    *,
    watermark_gray_threshold: int = 200,
    crop_box: tuple[int, int, int, int] | None = None,
    columns: int = 1,
    title: str = "Preprocessing stages",
) -> dict[str, Image.Image | np.ndarray]:
    """Display preprocessing stages and return them for closer inspection."""
    if crop_box is not None:
        image = image.crop(crop_box)

    stages = build_preprocessing_stages(
        image,
        watermark_gray_threshold=watermark_gray_threshold,
    )

    columns = max(1, min(columns, len(stages)))
    rows = int(np.ceil(len(stages) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(18 * columns, 8 * rows))
    fig.suptitle(title, fontsize=14)
    axes_array = np.atleast_1d(axes).ravel()
    for ax, (stage_name, stage_image) in zip(axes_array, stages.items()):
        if isinstance(stage_image, Image.Image):
            ax.imshow(stage_image)
        else:
            ax.imshow(stage_image, cmap="gray")
        ax.set_title(stage_name)
        ax.axis("off")

    for ax in axes_array[len(stages):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return stages
