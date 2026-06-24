from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def visualize_grid(
    grid_data: dict[str, Any],
    img: np.ndarray,
    line_width: int = 1,
    label_cells: bool = True,
) -> np.ndarray:
    """Draw a detected table grid over a grayscale/binary image."""
    if img.ndim == 2:
        result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        result_img = img.copy()

    y_positions = sorted({c["ymin"] for c in grid_data["cells"]} | {c["ymax"] for c in grid_data["cells"]})
    x_positions = sorted({c["xmin"] for c in grid_data["cells"]} | {c["xmax"] for c in grid_data["cells"]})

    for y in y_positions:
        cv2.line(result_img, (0, int(y)), (img.shape[1], int(y)), (0, 255, 255), line_width)
    for x in x_positions:
        cv2.line(result_img, (int(x), 0), (int(x), img.shape[0]), (255, 0, 0), line_width)

    for failed_line in grid_data.get("failed_horizontal", []):
        y = int(failed_line["y_center"])
        x_start = int(failed_line["x"])
        x_end = int(failed_line["x"] + failed_line["width"])
        _draw_dashed_line(result_img, (x_start, y), (x_end, y), (0, 165, 255), line_width + 1)

    for failed_line in grid_data.get("failed_vertical", []):
        x = int(failed_line["x_center"])
        y_start = int(failed_line["y"])
        y_end = int(failed_line["y"] + failed_line["height"])
        _draw_dashed_line(result_img, (x, y_start), (x, y_end), (0, 165, 255), line_width + 1)

    if label_cells:
        for i in range(len(y_positions) - 1):
            y_middle = int((y_positions[i] + y_positions[i + 1]) // 2)
            cv2.putText(result_img, f"H{i}", (10, y_middle), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        for i in range(len(x_positions) - 1):
            x_start = int(x_positions[i])
            cv2.putText(result_img, f"V{i}", (x_start, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    return result_img


def _draw_dashed_line(
    img: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_length: int = 20,
    gap_length: int = 10,
) -> None:
    x1, y1 = start
    x2, y2 = end
    if y1 == y2:
        x = x1
        while x < x2:
            cv2.line(img, (x, y1), (min(x + dash_length, x2), y2), color, thickness)
            x += dash_length + gap_length
        return

    y = y1
    while y < y2:
        cv2.line(img, (x1, y), (x2, min(y + dash_length, y2)), color, thickness)
        y += dash_length + gap_length

