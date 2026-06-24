from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .visualization import visualize_grid


GAUSSIAN_BLUR_KERNEL = (5, 5)
HORIZONTAL_KERNEL_DIVISOR = 6
VERTICAL_KERNEL_DIVISOR = 6
HORIZONTAL_MIN_LENGTH_PCT = 0.45
VERTICAL_MIN_LENGTH_PCT = 0.45
MAX_LINE_THICKNESS = 12
MIN_LINE_ASPECT_RATIO = 8
DILATION_ITERATIONS = 2
GRID_LINE_MERGE_TOLERANCE = 5
MIN_HORIZONTAL_LINE_SEPARATION_RATIO = 0.01
MIN_VERTICAL_LINE_SEPARATION_RATIO = 0.005
HORIZONTAL_DETECTOR = "combined"
HOUGH_THRESHOLD = 80
HOUGH_MIN_LINE_LENGTH_PCT = 0.40
HOUGH_MAX_LINE_GAP_PX = 25
HOUGH_MAX_ANGLE_DEG = 2.0
HOUGH_MERGE_TOLERANCE_PX = 5
HOUGH_MIN_WIDTH_COVERAGE = 0.70


def process_PIL_image_for_cv2(image: Image.Image, enhance_faint_lines: bool = False) -> np.ndarray:
    """Convert a PIL image into an inverted binary image for OpenCV line detection."""
    cv_image = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)

    if enhance_faint_lines:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cv_image = clahe.apply(cv_image)
        cv_image = cv2.fastNlMeansDenoising(cv_image, None, h=10, templateWindowSize=7, searchWindowSize=21)
        return cv2.adaptiveThreshold(
            cv_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=2,
        )

    cv_image = cv2.GaussianBlur(cv_image, GAUSSIAN_BLUR_KERNEL, 0)
    return cv2.threshold(cv_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]


def remove_light_watermark(image: Image.Image, gray_threshold: int = 200) -> Image.Image:
    """Whiten light gray watermark pixels while preserving dark table ink."""
    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Keep dark text/grid lines; remove mid/light gray watermark and background.
    cleaned = rgb.copy()
    cleaned[gray > gray_threshold] = 255
    return Image.fromarray(cleaned)


def detect_horizontal_vertical_lines(
    thres_img: np.ndarray,
    detect_horizontal: bool = True,
    detect_vertical: bool = True,
    horizontal_kernel_divisor: int = HORIZONTAL_KERNEL_DIVISOR,
    vertical_kernel_divisor: int = VERTICAL_KERNEL_DIVISOR,
    horizontal_min_length_pct: float = HORIZONTAL_MIN_LENGTH_PCT,
    vertical_min_length_pct: float = VERTICAL_MIN_LENGTH_PCT,
    max_thickness: int = MAX_LINE_THICKNESS,
    min_aspect_ratio: float = MIN_LINE_ASPECT_RATIO,
    merge_tolerance: int = GRID_LINE_MERGE_TOLERANCE,
    vertical_merge_tolerance: int | None = None,
    dilation_iterations: int = DILATION_ITERATIONS,
    horizontal_dilation_iterations: int | None = None,
    vertical_dilation_iterations: int | None = None,
    horizontal_detector: str = HORIZONTAL_DETECTOR,
) -> tuple[list[dict], list[dict], np.ndarray | None, np.ndarray | None, list[dict], list[dict], dict[str, Any] | None]:
    """Detect horizontal and vertical table lines from an inverted binary image."""
    horizontal_dilation_iterations = (
        dilation_iterations if horizontal_dilation_iterations is None else horizontal_dilation_iterations
    )
    vertical_dilation_iterations = (
        dilation_iterations if vertical_dilation_iterations is None else vertical_dilation_iterations
    )
    vertical_merge_tolerance = merge_tolerance if vertical_merge_tolerance is None else vertical_merge_tolerance

    if detect_horizontal:
        if horizontal_detector == "morphology":
            horizontal_lines, horizontal_img, failed_horizontal, horizontal_debug = extract_horizontal_lines(
                binary_img=thres_img,
                kernel_divisor=horizontal_kernel_divisor,
                min_length_pct=horizontal_min_length_pct,
                max_thickness=max_thickness,
                min_aspect_ratio=min_aspect_ratio,
                merge_tolerance=merge_tolerance,
                dilation_iterations=horizontal_dilation_iterations,
            )
        elif horizontal_detector == "hough":
            horizontal_lines, horizontal_img, failed_horizontal, horizontal_debug = extract_horizontal_lines_hough(
                binary_img=thres_img,
                min_length_pct=horizontal_min_length_pct,
                merge_tolerance=HOUGH_MERGE_TOLERANCE_PX,
                min_width_coverage=HOUGH_MIN_WIDTH_COVERAGE,
            )
        elif horizontal_detector == "combined":
            horizontal_lines, horizontal_img, failed_horizontal, horizontal_debug = extract_horizontal_lines_combined(
                binary_img=thres_img,
                kernel_divisor=horizontal_kernel_divisor,
                min_length_pct=horizontal_min_length_pct,
                max_thickness=max_thickness,
                min_aspect_ratio=min_aspect_ratio,
                merge_tolerance=merge_tolerance,
                dilation_iterations=horizontal_dilation_iterations,
            )
        else:
            raise ValueError(
                "horizontal_detector must be one of: 'morphology', 'hough', 'combined'"
            )
    else:
        horizontal_lines, horizontal_img, failed_horizontal, horizontal_debug = [], None, [], None

    if detect_vertical:
        vertical_lines, vertical_img, failed_vertical = extract_vertical_lines(
            binary_img=thres_img,
            kernel_divisor=vertical_kernel_divisor,
            min_length_pct=vertical_min_length_pct,
            max_thickness=max_thickness,
            min_aspect_ratio=min_aspect_ratio,
            merge_tolerance=vertical_merge_tolerance,
            dilation_iterations=vertical_dilation_iterations,
        )
    else:
        vertical_lines, vertical_img, failed_vertical = [], None, []

    return horizontal_lines, vertical_lines, horizontal_img, vertical_img, failed_horizontal, failed_vertical, horizontal_debug


def create_grid(
    horizontal_lines: list[dict],
    vertical_lines: list[dict],
    img: np.ndarray,
    line_width: int = 1,
    failed_horizontal: list[dict] | None = None,
    failed_vertical: list[dict] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Create rectangular cell boxes from detected horizontal and vertical lines."""
    img_height, img_width = img.shape[:2]
    y_positions = [0] + [line["y_center"] for line in horizontal_lines] + [img_height]
    x_positions = [0] + [line["x_center"] for line in vertical_lines] + [img_width]

    cells = []
    for row_idx in range(len(y_positions) - 1):
        for col_idx in range(len(x_positions) - 1):
            cells.append(
                {
                    "row": row_idx,
                    "col": col_idx,
                    "row_name": f"H{row_idx}",
                    "col_name": f"V{col_idx}",
                    "xmin": int(x_positions[col_idx]),
                    "xmax": int(x_positions[col_idx + 1]),
                    "ymin": int(y_positions[row_idx]),
                    "ymax": int(y_positions[row_idx + 1]),
                    "width": int(x_positions[col_idx + 1] - x_positions[col_idx]),
                    "height": int(y_positions[row_idx + 1] - y_positions[row_idx]),
                }
            )

    rows = len(y_positions) - 1
    cols = len(x_positions) - 1
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for cell in cells:
        grid[cell["row"]][cell["col"]] = cell

    grid_data = {
        "cells": cells,
        "rows": rows,
        "cols": cols,
        "grid": grid,
        "failed_horizontal": failed_horizontal or [],
        "failed_vertical": failed_vertical or [],
    }
    return grid_data, visualize_grid(grid_data, img, line_width=line_width)


def infer_table_bbox(
    horizontal_lines: list[dict],
    vertical_lines: list[dict],
    image_size: tuple[int, int],
    padding: int = 4,
) -> dict[str, int]:
    """Infer the outer table bbox from accepted full-page table lines."""
    if not horizontal_lines:
        raise ValueError("Cannot infer table bbox without horizontal lines.")
    if not vertical_lines:
        raise ValueError("Cannot infer table bbox without vertical lines.")

    image_width, image_height = image_size
    xmin = min(line["x_center"] for line in vertical_lines) - padding
    xmax = max(line["x_center"] for line in vertical_lines) + padding
    ymin = min(min(line.get("y", line["y_center"]), line["y_center"]) for line in horizontal_lines) - padding
    ymax = max(
        max(line.get("y", line["y_center"]) + line.get("height", 1), line["y_center"])
        for line in horizontal_lines
    ) + padding

    return {
        "xmin": max(0, int(xmin)),
        "ymin": max(0, int(ymin)),
        "xmax": min(image_width, int(xmax)),
        "ymax": min(image_height, int(ymax)),
    }


def crop_pil_to_bbox(image: Image.Image, bbox: dict[str, int]) -> Image.Image:
    """Crop a PIL image using a dict bbox with xmin, ymin, xmax, ymax."""
    return image.crop((bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]))


def drop_lines_near_image_boundary(
    horizontal_lines: list[dict],
    vertical_lines: list[dict],
    image_shape: tuple[int, int],
    tolerance: int = 8,
) -> tuple[list[dict], list[dict]]:
    """Drop detected border lines near crop edges, leaving internal separators."""
    img_height, img_width = image_shape[:2]
    inner_h = [
        line for line in horizontal_lines
        if tolerance < line["y_center"] < img_height - tolerance
    ]
    inner_v = [
        line for line in vertical_lines
        if tolerance < line["x_center"] < img_width - tolerance
    ]
    return inner_h, inner_v


def get_row_bboxs(grid_data: dict[str, Any], grid_image: np.ndarray) -> list[dict]:
    cell_df = pd.DataFrame(grid_data["cells"])
    rows = cell_df[["row", "ymin", "ymax"]].drop_duplicates()
    rows["xmin"] = 0
    rows["xmax"] = grid_image.shape[1]
    return rows.to_dict(orient="records")


def get_column_bboxs(grid_data: dict[str, Any], grid_image: np.ndarray) -> list[dict]:
    cell_df = pd.DataFrame(grid_data["cells"])
    columns = cell_df[["col", "xmin", "xmax"]].drop_duplicates()
    columns["ymin"] = 0
    columns["ymax"] = grid_image.shape[0]
    return columns.to_dict(orient="records")


def extract_horizontal_lines(
    binary_img: np.ndarray,
    kernel_divisor: int = HORIZONTAL_KERNEL_DIVISOR,
    min_length_pct: float = HORIZONTAL_MIN_LENGTH_PCT,
    max_thickness: int = MAX_LINE_THICKNESS,
    min_aspect_ratio: float = MIN_LINE_ASPECT_RATIO,
    merge_tolerance: int = GRID_LINE_MERGE_TOLERANCE,
    dilation_iterations: int = DILATION_ITERATIONS,
) -> tuple[list[dict], np.ndarray, list[dict], dict[str, Any]]:
    img_height, img_width = binary_img.shape
    kernel_len = max(1, img_width // kernel_divisor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))

    horizontal = cv2.erode(binary_img.copy(), kernel, iterations=1)
    horizontal = cv2.dilate(horizontal, kernel, iterations=dilation_iterations)
    morph_img = horizontal.copy()
    contours, _ = cv2.findContours(horizontal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_length_px = int(img_width * min_length_pct)
    candidate_lines: list[dict] = []
    failed_lines: list[dict] = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        if h <= max_thickness and aspect_ratio >= min_aspect_ratio:
            candidate_lines.append({"x": x, "y": y, "width": w, "height": h, "y_center": y + h // 2})
        else:
            failed_lines.append(
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "y_center": y + h // 2,
                    "reason": "thickness" if h > max_thickness else "aspect",
                }
            )

    candidate_lines = sorted(candidate_lines, key=lambda line: line["y_center"])
    candidate_lines_pre_merge = [line.copy() for line in candidate_lines]
    if merge_tolerance > 0:
        candidate_lines = merge_close_lines(candidate_lines, merge_tolerance, axis="horizontal")
    candidate_lines_post_merge = [line.copy() for line in candidate_lines]

    lines = []
    length_failed = []
    for line in candidate_lines:
        if line["width"] >= min_length_px:
            lines.append(line)
        else:
            length_failed.append({**line, "reason": "length"})

    min_separation_px = int(img_height * MIN_HORIZONTAL_LINE_SEPARATION_RATIO)
    filtered_lines = []
    proximity_removed = []
    for line in sorted(lines, key=lambda item: item["y_center"]):
        if filtered_lines and line["y_center"] - filtered_lines[-1]["y_center"] < min_separation_px:
            proximity_removed.append({**line, "reason": "proximity"})
            continue
        filtered_lines.append(line)

    boundary_removed = []
    boundary_filtered = []
    for line in filtered_lines:
        if line["y_center"] < min_separation_px:
            boundary_removed.append({**line, "reason": "boundary_top"})
            continue
        if img_height - line["y_center"] < min_separation_px:
            boundary_removed.append({**line, "reason": "boundary_bottom"})
            continue
        boundary_filtered.append(line)

    lines = sorted(boundary_filtered, key=lambda line: line["y_center"])
    failed_lines = sorted(failed_lines + length_failed + proximity_removed + boundary_removed, key=lambda line: line["y_center"])

    filtered_img = np.zeros_like(binary_img)
    for line in lines:
        filtered_img[line["y"] : line["y"] + line["height"], line["x"] : line["x"] + line["width"]] = 255

    debug_info = {
        "morph_img": morph_img,
        "candidate_lines_pre_merge": candidate_lines_pre_merge,
        "candidate_lines_post_merge": candidate_lines_post_merge,
        "length_failed": [line.copy() for line in length_failed],
        "proximity_removed": [line.copy() for line in proximity_removed],
        "boundary_removed": [line.copy() for line in boundary_removed],
        "final_lines": [line.copy() for line in lines],
        "thresholds": {
            "img_width": img_width,
            "img_height": img_height,
            "kernel_len": kernel_len,
            "min_length_px": min_length_px,
            "max_thickness": max_thickness,
            "min_aspect_ratio": min_aspect_ratio,
            "merge_tolerance": merge_tolerance,
            "min_separation_px": min_separation_px,
        },
    }
    return lines, filtered_img, failed_lines, debug_info


def extract_horizontal_lines_hough(
    binary_img: np.ndarray,
    min_length_pct: float = HORIZONTAL_MIN_LENGTH_PCT,
    threshold: int = HOUGH_THRESHOLD,
    max_line_gap_px: int = HOUGH_MAX_LINE_GAP_PX,
    max_angle_deg: float = HOUGH_MAX_ANGLE_DEG,
    merge_tolerance: int = HOUGH_MERGE_TOLERANCE_PX,
    min_width_coverage: float = HOUGH_MIN_WIDTH_COVERAGE,
    coverage_band_px: int = 2,
) -> tuple[list[dict], np.ndarray, list[dict], dict[str, Any]]:
    """Detect horizontal lines with probabilistic Hough transform."""
    img_height, img_width = binary_img.shape
    min_line_length_px = max(1, int(img_width * min_length_pct))
    segments = cv2.HoughLinesP(
        binary_img,
        rho=1,
        theta=np.pi / 180,
        threshold=threshold,
        minLineLength=min_line_length_px,
        maxLineGap=max_line_gap_px,
    )

    raw_segments = [] if segments is None else [seg[0] for seg in segments]
    candidates = []
    failed_lines = []
    max_angle_rad = np.deg2rad(max_angle_deg)

    for x1, y1, x2, y2 in raw_segments:
        angle = np.arctan2(abs(int(y2) - int(y1)), max(1, abs(int(x2) - int(x1))))
        x_min = int(min(x1, x2))
        x_max = int(max(x1, x2))
        y_center = int(round((int(y1) + int(y2)) / 2))
        rec = {
            "x": x_min,
            "y": y_center,
            "width": x_max - x_min,
            "height": 1,
            "y_center": y_center,
            "source": "hough",
        }
        if angle <= max_angle_rad:
            candidates.append(rec)
        else:
            failed_lines.append({**rec, "reason": "angle", "angle_deg": float(np.rad2deg(angle))})

    candidates = merge_close_lines(sorted(candidates, key=lambda line: line["y_center"]), merge_tolerance, axis="horizontal")

    lines = []
    low_coverage = []
    min_separation_px = int(img_height * MIN_HORIZONTAL_LINE_SEPARATION_RATIO)
    for line in candidates:
        coverage = _measure_width_coverage(binary_img, line["y_center"], coverage_band_px)
        if coverage >= min_width_coverage:
            lines.append({**line, "coverage": coverage})
        else:
            low_coverage.append({**line, "reason": "coverage", "coverage": coverage})

    filtered_lines = []
    proximity_removed = []
    for line in sorted(lines, key=lambda item: item["y_center"]):
        if filtered_lines and line["y_center"] - filtered_lines[-1]["y_center"] < min_separation_px:
            proximity_removed.append({**line, "reason": "proximity"})
            continue
        filtered_lines.append(line)

    lines = sorted(filtered_lines, key=lambda line: line["y_center"])
    failed_lines = sorted(failed_lines + low_coverage + proximity_removed, key=lambda line: line["y_center"])

    filtered_img = np.zeros_like(binary_img)
    for line in lines:
        y = int(line["y_center"])
        x0 = max(0, int(line["x"]))
        x1 = min(img_width, int(line["x"] + line["width"]))
        filtered_img[max(0, y - 1): min(img_height, y + 2), x0:x1] = 255

    debug_info = {
        "detector": "hough",
        "raw_segment_count": len(raw_segments),
        "candidate_lines": candidates,
        "final_lines": [line.copy() for line in lines],
        "failed_lines": [line.copy() for line in failed_lines],
        "thresholds": {
            "threshold": threshold,
            "min_line_length_px": min_line_length_px,
            "max_line_gap_px": max_line_gap_px,
            "max_angle_deg": max_angle_deg,
            "merge_tolerance": merge_tolerance,
            "min_width_coverage": min_width_coverage,
        },
    }
    return lines, filtered_img, failed_lines, debug_info


def extract_horizontal_lines_combined(
    binary_img: np.ndarray,
    kernel_divisor: int = HORIZONTAL_KERNEL_DIVISOR,
    min_length_pct: float = HORIZONTAL_MIN_LENGTH_PCT,
    max_thickness: int = MAX_LINE_THICKNESS,
    min_aspect_ratio: float = MIN_LINE_ASPECT_RATIO,
    merge_tolerance: int = GRID_LINE_MERGE_TOLERANCE,
    dilation_iterations: int = DILATION_ITERATIONS,
    agreement_tolerance: int = HOUGH_MERGE_TOLERANCE_PX,
) -> tuple[list[dict], np.ndarray, list[dict], dict[str, Any]]:
    """Combine morphology and Hough horizontal-line detections."""
    morph_lines, morph_img, morph_failed, morph_debug = extract_horizontal_lines(
        binary_img=binary_img,
        kernel_divisor=kernel_divisor,
        min_length_pct=min_length_pct,
        max_thickness=max_thickness,
        min_aspect_ratio=min_aspect_ratio,
        merge_tolerance=merge_tolerance,
        dilation_iterations=dilation_iterations,
    )
    hough_lines, hough_img, hough_failed, hough_debug = extract_horizontal_lines_hough(
        binary_img=binary_img,
        min_length_pct=min_length_pct,
        merge_tolerance=agreement_tolerance,
        min_width_coverage=HOUGH_MIN_WIDTH_COVERAGE,
    )

    combined_lines = []
    matched_hough_indexes = set()

    for morph_line in morph_lines:
        match_idx = _find_nearest_line_by_y(morph_line, hough_lines, agreement_tolerance, matched_hough_indexes)
        if match_idx is None:
            combined_lines.append({**morph_line, "source": "morphology_only"})
            continue

        matched_hough_indexes.add(match_idx)
        hough_line = hough_lines[match_idx]
        combined_lines.append(_merge_horizontal_line_pair(morph_line, hough_line, source="morphology+hough"))

    for i, hough_line in enumerate(hough_lines):
        if i in matched_hough_indexes:
            continue
        combined_lines.append({**hough_line, "source": "hough_only"})

    combined_lines = sorted(combined_lines, key=lambda line: line["y_center"])
    combined_lines = merge_close_lines(combined_lines, merge_tolerance, axis="horizontal")
    for line in combined_lines:
        line.setdefault("source", "combined")

    img_height, img_width = binary_img.shape
    combined_img = np.zeros_like(binary_img)
    for line in combined_lines:
        y0 = max(0, int(line["y"]))
        y1 = min(img_height, int(line["y"] + max(1, line["height"])))
        x0 = max(0, int(line["x"]))
        x1 = min(img_width, int(line["x"] + line["width"]))
        combined_img[y0:y1, x0:x1] = 255

    debug_info = {
        "detector": "combined",
        "morphology": morph_debug,
        "hough": hough_debug,
        "final_lines": [line.copy() for line in combined_lines],
        "morphology_count": len(morph_lines),
        "hough_count": len(hough_lines),
        "combined_count": len(combined_lines),
        "source_counts": _count_by_source(combined_lines),
    }
    failed_lines = sorted(morph_failed + hough_failed, key=lambda line: line["y_center"])
    return combined_lines, combined_img, failed_lines, debug_info


def extract_vertical_lines(
    binary_img: np.ndarray,
    kernel_divisor: int = VERTICAL_KERNEL_DIVISOR,
    min_length_pct: float = VERTICAL_MIN_LENGTH_PCT,
    max_thickness: int = MAX_LINE_THICKNESS,
    min_aspect_ratio: float = MIN_LINE_ASPECT_RATIO,
    merge_tolerance: int = GRID_LINE_MERGE_TOLERANCE,
    dilation_iterations: int = DILATION_ITERATIONS,
) -> tuple[list[dict], np.ndarray, list[dict]]:
    img_height, img_width = binary_img.shape
    kernel_len = max(1, img_height // kernel_divisor)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))

    vertical = cv2.erode(binary_img.copy(), kernel, iterations=1)
    vertical = cv2.dilate(vertical, kernel, iterations=dilation_iterations)
    contours, _ = cv2.findContours(vertical, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_length_px = int(img_height * min_length_pct)
    candidate_lines = []
    failed_lines = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = h / w if w > 0 else 0
        if w <= max_thickness and aspect_ratio >= min_aspect_ratio:
            candidate_lines.append({"x": x, "y": y, "width": w, "height": h, "x_center": x + w // 2})
        else:
            failed_lines.append(
                {
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "x_center": x + w // 2,
                    "reason": "thickness" if w > max_thickness else "aspect",
                }
            )

    candidate_lines = sorted(candidate_lines, key=lambda line: line["x_center"])
    if merge_tolerance > 0:
        candidate_lines = merge_close_lines(candidate_lines, merge_tolerance, axis="vertical")

    lines = []
    length_failed = []
    for line in candidate_lines:
        if line["height"] >= min_length_px:
            lines.append(line)
        else:
            length_failed.append({**line, "reason": "length"})

    min_separation_px = int(img_width * MIN_VERTICAL_LINE_SEPARATION_RATIO)
    filtered_lines = []
    proximity_removed = []
    lines_sorted = sorted(lines, key=lambda line: line["x_center"])
    for i, line in enumerate(lines_sorted):
        if i < len(lines_sorted) - 1:
            distance = lines_sorted[i + 1]["x_center"] - line["x_center"]
            if distance < min_separation_px:
                proximity_removed.append({**line, "reason": "proximity"})
                continue
        filtered_lines.append(line)

    lines = sorted(filtered_lines, key=lambda line: line["x_center"])
    failed_lines = sorted(failed_lines + length_failed + proximity_removed, key=lambda line: line["x_center"])

    filtered_img = np.zeros_like(binary_img)
    for line in lines:
        filtered_img[line["y"] : line["y"] + line["height"], line["x"] : line["x"] + line["width"]] = 255

    return lines, filtered_img, failed_lines


def merge_close_lines(lines: list[dict], tolerance: int, axis: str = "horizontal") -> list[dict]:
    """Merge near-duplicate line detections caused by thick or double rule edges."""
    if not lines or tolerance <= 0:
        return lines

    center_key = "y_center" if axis == "horizontal" else "x_center"
    sorted_lines = sorted(lines, key=lambda line: line[center_key])
    merged = [sorted_lines[0].copy()]

    for line in sorted_lines[1:]:
        last_line = merged[-1]
        if abs(line[center_key] - last_line[center_key]) <= tolerance:
            if "source" in last_line or "source" in line:
                last_line["source"] = _merge_source_labels(
                    last_line.get("source"),
                    line.get("source"),
                )
            last_line[center_key] = line[center_key]
            if axis == "horizontal":
                x_min = min(last_line["x"], line["x"])
                x_max = max(last_line["x"] + last_line["width"], line["x"] + line["width"])
                last_line["x"] = x_min
                last_line["width"] = x_max - x_min
            else:
                y_min = min(last_line["y"], line["y"])
                y_max = max(last_line["y"] + last_line["height"], line["y"] + line["height"])
                last_line["y"] = y_min
                last_line["height"] = y_max - y_min
        else:
            merged.append(line.copy())

    return merged


def _measure_width_coverage(binary_img: np.ndarray, y_center: int, band_px: int) -> float:
    """Measure left-to-right ink span around a horizontal candidate."""
    img_h, img_w = binary_img.shape
    if img_w == 0:
        return 0.0
    y_lo = max(0, int(y_center) - band_px)
    y_hi = min(img_h, int(y_center) + band_px + 1)
    if y_hi <= y_lo:
        return 0.0

    column_has_ink = binary_img[y_lo:y_hi, :].any(axis=0)
    if not column_has_ink.any():
        return 0.0

    x_left = int(np.argmax(column_has_ink))
    x_right = int(img_w - 1 - np.argmax(column_has_ink[::-1]))
    return (x_right - x_left + 1) / img_w


def _find_nearest_line_by_y(
    line: dict,
    candidates: list[dict],
    tolerance: int,
    already_used: set[int],
) -> int | None:
    best_idx = None
    best_distance = tolerance + 1
    for idx, candidate in enumerate(candidates):
        if idx in already_used:
            continue
        distance = abs(int(line["y_center"]) - int(candidate["y_center"]))
        if distance <= tolerance and distance < best_distance:
            best_idx = idx
            best_distance = distance
    return best_idx


def _merge_horizontal_line_pair(morph_line: dict, hough_line: dict, source: str) -> dict:
    x_min = min(int(morph_line["x"]), int(hough_line["x"]))
    x_max = max(
        int(morph_line["x"] + morph_line["width"]),
        int(hough_line["x"] + hough_line["width"]),
    )
    y_center = int(round((int(morph_line["y_center"]) + int(hough_line["y_center"])) / 2))
    height = max(1, int(morph_line.get("height", 1)), int(hough_line.get("height", 1)))
    return {
        **morph_line,
        "x": x_min,
        "y": int(y_center - height // 2),
        "width": x_max - x_min,
        "height": height,
        "y_center": y_center,
        "source": source,
        "hough_coverage": hough_line.get("coverage"),
    }


def _merge_source_labels(left: str | None, right: str | None) -> str:
    labels = {label for label in (left, right) if label}
    if not labels:
        return "combined"
    if "morphology+hough" in labels:
        return "morphology+hough"
    if labels == {"morphology_only", "hough_only"}:
        return "morphology+hough"
    return "+".join(sorted(labels))


def _count_by_source(lines: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in lines:
        source = line.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts
