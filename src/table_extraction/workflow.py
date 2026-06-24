from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
from PIL import Image

from .grid_detection import (
    create_grid,
    crop_pil_to_bbox,
    detect_horizontal_vertical_lines,
    drop_lines_near_image_boundary,
    infer_table_bbox,
    process_PIL_image_for_cv2,
    remove_light_watermark,
)


@dataclass
class TableBoundaryResult:
    page_path: Path
    page_image: Image.Image
    cleaned_page_image: Image.Image
    full_binary_image: np.ndarray
    full_h_lines: list[dict]
    full_v_lines: list[dict]
    full_failed_h: list[dict]
    full_failed_v: list[dict]
    full_h_debug: dict
    full_line_overlay: np.ndarray
    table_bbox: dict[str, int]
    table_image: Image.Image
    table_binary_image: np.ndarray


@dataclass
class TableGridResult(TableBoundaryResult):
    h_lines: list[dict]
    v_lines: list[dict]
    inner_h_lines: list[dict]
    inner_v_lines: list[dict]
    failed_h: list[dict]
    failed_v: list[dict]
    h_debug: dict
    grid_data: dict
    grid_overlay: np.ndarray


def detect_table_boundary_from_page(
    page_path: Path,
    *,
    bbox_padding: int = 4,
    remove_watermark: bool = True,
    watermark_gray_threshold: int = 200,
) -> TableBoundaryResult:
    """Detect full-page lines, infer the table bbox, and crop the table."""
    page_path = Path(page_path)
    page_image = Image.open(page_path).convert("RGB")
    cleaned_page_image = (
        remove_light_watermark(page_image, gray_threshold=watermark_gray_threshold)
        if remove_watermark
        else page_image
    )
    full_binary_image = process_PIL_image_for_cv2(cleaned_page_image)

    full_h_lines, full_v_lines, _, _, full_failed_h, full_failed_v, full_h_debug = detect_horizontal_vertical_lines(
        full_binary_image
    )
    _, full_line_overlay = create_grid(
        full_h_lines,
        full_v_lines,
        full_binary_image,
        line_width=2,
        failed_horizontal=full_failed_h,
        failed_vertical=full_failed_v,
    )

    table_bbox = infer_table_bbox(full_h_lines, full_v_lines, page_image.size, padding=bbox_padding)
    table_image = crop_pil_to_bbox(cleaned_page_image, table_bbox)
    table_binary_image = process_PIL_image_for_cv2(table_image)

    return TableBoundaryResult(
        page_path=page_path,
        page_image=page_image,
        cleaned_page_image=cleaned_page_image,
        full_binary_image=full_binary_image,
        full_h_lines=full_h_lines,
        full_v_lines=full_v_lines,
        full_failed_h=full_failed_h,
        full_failed_v=full_failed_v,
        full_h_debug=full_h_debug or {},
        full_line_overlay=full_line_overlay,
        table_bbox=table_bbox,
        table_image=table_image,
        table_binary_image=table_binary_image,
    )


def detect_grid_from_table_crop(
    boundary_result: TableBoundaryResult,
    *,
    boundary_tolerance: int = 8,
) -> TableGridResult:
    """Detect the internal grid inside an already-cropped table."""
    table_binary_image = boundary_result.table_binary_image
    h_lines, v_lines, _, _, failed_h, failed_v, h_debug = detect_horizontal_vertical_lines(table_binary_image)
    inner_h_lines, inner_v_lines = drop_lines_near_image_boundary(
        h_lines,
        v_lines,
        table_binary_image.shape,
        tolerance=boundary_tolerance,
    )
    grid_data, grid_overlay = create_grid(
        inner_h_lines,
        inner_v_lines,
        table_binary_image,
        line_width=2,
        failed_horizontal=failed_h,
        failed_vertical=failed_v,
    )

    return TableGridResult(
        page_path=boundary_result.page_path,
        page_image=boundary_result.page_image,
        cleaned_page_image=boundary_result.cleaned_page_image,
        full_binary_image=boundary_result.full_binary_image,
        full_h_lines=boundary_result.full_h_lines,
        full_v_lines=boundary_result.full_v_lines,
        full_failed_h=boundary_result.full_failed_h,
        full_failed_v=boundary_result.full_failed_v,
        full_h_debug=boundary_result.full_h_debug,
        full_line_overlay=boundary_result.full_line_overlay,
        table_bbox=boundary_result.table_bbox,
        table_image=boundary_result.table_image,
        table_binary_image=boundary_result.table_binary_image,
        h_lines=h_lines,
        v_lines=v_lines,
        inner_h_lines=inner_h_lines,
        inner_v_lines=inner_v_lines,
        failed_h=failed_h,
        failed_v=failed_v,
        h_debug=h_debug or {},
        grid_data=grid_data,
        grid_overlay=grid_overlay,
    )


def detect_table_grid_from_page(
    page_path: Path,
    *,
    bbox_padding: int = 4,
    boundary_tolerance: int = 8,
    remove_watermark: bool = True,
    watermark_gray_threshold: int = 200,
) -> TableGridResult:
    """Convenience wrapper for boundary detection followed by grid detection."""
    boundary_result = detect_table_boundary_from_page(
        page_path,
        bbox_padding=bbox_padding,
        remove_watermark=remove_watermark,
        watermark_gray_threshold=watermark_gray_threshold,
    )
    return detect_grid_from_table_crop(boundary_result, boundary_tolerance=boundary_tolerance)


def print_table_boundary_summary(result: TableBoundaryResult) -> None:
    """Print full-page line-detection and table-boundary counts."""
    print(f"Full-page H lines: {len(result.full_h_lines)}")
    print(f"Full-page V lines: {len(result.full_v_lines)}")
    print(f"Rejected H candidates: {len(result.full_failed_h)}")
    print(f"Rejected V candidates: {len(result.full_failed_v)}")
    if result.full_h_debug.get("source_counts"):
        print(f"Full-page H sources: {result.full_h_debug['source_counts']}")
    print(f"Table bbox: {result.table_bbox}")
    print(f"Cropped table size: {result.table_image.size}")


def print_table_grid_summary(result: TableGridResult) -> None:
    """Print cropped-table line-detection and grid counts."""
    print(f"Cropped-table H lines: {len(result.h_lines)} -> internal: {len(result.inner_h_lines)}")
    print(f"Cropped-table V lines: {len(result.v_lines)} -> internal: {len(result.inner_v_lines)}")
    if result.h_debug.get("source_counts"):
        print(f"Cropped-table H sources: {result.h_debug['source_counts']}")
    print(f"Grid rows: {result.grid_data['rows']}")
    print(f"Grid cols: {result.grid_data['cols']}")


def full_page_line_audit(result: TableBoundaryResult) -> pd.DataFrame:
    """Return accepted/rejected full-page line candidates with source and reason."""
    return _build_line_audit_table(
        accepted_horizontal=result.full_h_lines,
        accepted_vertical=result.full_v_lines,
        rejected_horizontal=result.full_failed_h,
        rejected_vertical=result.full_failed_v,
        scope="full_page",
    )


def cropped_grid_line_audit(result: TableGridResult) -> pd.DataFrame:
    """Return accepted/rejected cropped-table grid candidates with source and reason."""
    return _build_line_audit_table(
        accepted_horizontal=result.inner_h_lines,
        accepted_vertical=result.inner_v_lines,
        rejected_horizontal=result.failed_h,
        rejected_vertical=result.failed_v,
        scope="cropped_table",
    )


def show_full_page_line_detection(result: TableBoundaryResult) -> None:
    """Display accepted and rejected full-page line candidates."""
    show_line_detection_overlay(
        image=result.cleaned_page_image,
        horizontal_lines=result.full_h_lines,
        vertical_lines=result.full_v_lines,
        failed_horizontal=result.full_failed_h,
        failed_vertical=result.full_failed_v,
        title=f"Full-page detected lines: {result.page_path.name}",
    )


def show_horizontal_detector_comparison(
    result: TableBoundaryResult | TableGridResult,
    scope: str = "full_page",
    *,
    label_sources: bool = False,
) -> None:
    """Display morphology, Hough, and combined horizontal detections side by side."""
    if scope == "full_page":
        image = result.full_binary_image
        debug = result.full_h_debug
        title_prefix = "Full page"
    elif scope == "cropped_table":
        image = result.table_binary_image
        debug = result.h_debug
        title_prefix = "Cropped table"
    else:
        raise ValueError("scope must be 'full_page' or 'cropped_table'")

    morph_lines = (debug.get("morphology") or {}).get("final_lines", [])
    hough_lines = (debug.get("hough") or {}).get("final_lines", [])
    combined_lines = debug.get("final_lines", [])
    if not combined_lines:
        combined_lines = result.full_h_lines if scope == "full_page" else result.h_lines

    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=True, sharey=True)
    panels = [
        ("Morphology", morph_lines, "lime"),
        ("Hough", hough_lines, "cyan"),
        ("Combined", combined_lines, "yellow"),
    ]

    for ax, (label, lines, color) in zip(axes, panels):
        ax.imshow(image, cmap="gray")
        _draw_horizontal_lines(ax, lines, color=color, label_sources=label_sources)
        ax.set_title(f"{title_prefix}: {label} ({len(lines)} lines)")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def show_table_boundary(result: TableBoundaryResult) -> None:
    """Display the inferred table bbox and cropped binary table image."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].imshow(result.cleaned_page_image)
    axes[0].add_patch(
        plt.Rectangle(
            (result.table_bbox["xmin"], result.table_bbox["ymin"]),
            result.table_bbox["xmax"] - result.table_bbox["xmin"],
            result.table_bbox["ymax"] - result.table_bbox["ymin"],
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
    axes[0].set_title("Inferred outer table boundary")
    axes[0].axis("off")

    axes[1].imshow(result.table_binary_image, cmap="gray")
    axes[1].set_title("Cropped table binary image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_cropped_table_grid(result: TableGridResult) -> None:
    """Display the internal grid detected inside the cropped table."""
    show_line_detection_overlay(
        image=result.table_image,
        horizontal_lines=result.inner_h_lines,
        vertical_lines=result.inner_v_lines,
        failed_horizontal=result.failed_h,
        failed_vertical=result.failed_v,
        title=(
            f"Internal grid inside cropped table: "
            f"{result.grid_data['rows']} rows x {result.grid_data['cols']} cols"
        ),
    )


def show_cropped_table_grid_review(result: TableGridResult) -> None:
    """Display the cropped table and final grid overlay for notebook review."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(result.table_image)
    axes[0].set_title("Cropped table")
    axes[0].axis("off")

    axes[1].imshow(cv2.cvtColor(result.grid_overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(
        f"Final internal grid: {result.grid_data['rows']} rows x {result.grid_data['cols']} cols"
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def show_line_detection_overlay(
    *,
    image: Image.Image | np.ndarray,
    horizontal_lines: list[dict],
    vertical_lines: list[dict],
    failed_horizontal: list[dict] | None = None,
    failed_vertical: list[dict] | None = None,
    title: str = "Detected lines",
) -> None:
    """Shared line-overlay plot with consistent colors and legend."""
    failed_horizontal = failed_horizontal or []
    failed_vertical = failed_vertical or []

    fig, ax = plt.subplots(figsize=(16, 10))
    if isinstance(image, Image.Image):
        ax.imshow(image)
    elif image.ndim == 2:
        ax.imshow(image, cmap="gray")
    else:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    _draw_horizontal_lines(ax, horizontal_lines, color="#d000ff", linewidth=1.8)
    _draw_vertical_lines(ax, vertical_lines, color="#00a3ff", linewidth=1.8)
    _draw_horizontal_lines(ax, failed_horizontal, color="#ff9f1c", linewidth=1.2, linestyle="--")
    _draw_vertical_lines(ax, failed_vertical, color="#ff4d4d", linewidth=1.2, linestyle="--")

    handles = [
        mlines.Line2D([], [], color="#d000ff", linewidth=2, label=f"Accepted horizontal ({len(horizontal_lines)})"),
        mlines.Line2D([], [], color="#00a3ff", linewidth=2, label=f"Accepted vertical ({len(vertical_lines)})"),
        mlines.Line2D([], [], color="#ff9f1c", linewidth=2, linestyle="--", label=f"Rejected horizontal ({len(failed_horizontal)})"),
        mlines.Line2D([], [], color="#ff4d4d", linewidth=2, linestyle="--", label=f"Rejected vertical ({len(failed_vertical)})"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9, fontsize=10)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def _build_line_audit_table(
    *,
    accepted_horizontal: list[dict],
    accepted_vertical: list[dict],
    rejected_horizontal: list[dict],
    rejected_vertical: list[dict],
    scope: str,
) -> pd.DataFrame:
    rows = []
    for axis, status, lines in [
        ("horizontal", "accepted", accepted_horizontal),
        ("vertical", "accepted", accepted_vertical),
        ("horizontal", "rejected", rejected_horizontal),
        ("vertical", "rejected", rejected_vertical),
    ]:
        for index, line in enumerate(lines):
            rows.append(_line_audit_record(line, scope=scope, axis=axis, status=status, index=index))

    columns = [
        "scope",
        "axis",
        "status",
        "index",
        "source",
        "reason",
        "x",
        "y",
        "x_center",
        "y_center",
        "width",
        "height",
        "coverage",
        "angle_deg",
    ]
    table = pd.DataFrame(rows, columns=columns)
    if table.empty:
        return table
    return table.sort_values(["status", "axis", "y_center", "x_center"], na_position="last").reset_index(drop=True)


def _line_audit_record(line: dict, *, scope: str, axis: str, status: str, index: int) -> dict:
    return {
        "scope": scope,
        "axis": axis,
        "status": status,
        "index": index,
        "source": _line_source(line, axis=axis),
        "reason": line.get("reason", "") if status == "rejected" else "",
        "x": line.get("x"),
        "y": line.get("y"),
        "x_center": line.get("x_center"),
        "y_center": line.get("y_center"),
        "width": line.get("width"),
        "height": line.get("height"),
        "coverage": line.get("coverage", line.get("hough_coverage")),
        "angle_deg": line.get("angle_deg"),
    }


def _line_source(line: dict, *, axis: str) -> str:
    source = line.get("source")
    if source:
        return source
    if axis == "vertical":
        return "morphology"
    return "unknown"


def _draw_horizontal_lines(
    ax,
    lines: list[dict],
    color: str,
    label_sources: bool = False,
    linewidth: float = 2,
    linestyle: str = "-",
) -> None:
    for line in lines:
        y = line["y_center"]
        x0 = line.get("x", 0)
        x1 = x0 + line.get("width", 0)
        ax.plot([x0, x1], [y, y], color=color, linewidth=linewidth, linestyle=linestyle)
        source = line.get("source")
        if label_sources and source:
            ax.text(x1 + 4, y, source, color=color, fontsize=7, va="center")


def _draw_vertical_lines(
    ax,
    lines: list[dict],
    color: str,
    linewidth: float = 2,
    linestyle: str = "-",
) -> None:
    for line in lines:
        x = line["x_center"]
        y0 = line.get("y", 0)
        y1 = y0 + line.get("height", 0)
        ax.plot([x, x], [y0, y1], color=color, linewidth=linewidth, linestyle=linestyle)
